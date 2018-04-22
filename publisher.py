#!/usr/bin/env python3

import argparse
import docker
import yaml
import logging
import re
import jinja2

from requests.auth import HTTPDigestAuth, HTTPBasicAuth
from docker_registry_util import client
from typing import Dict, List, Optional

LOG = logging.getLogger("publisher")
LOG.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOG.addHandler(ch)

class Image(object):
    def __init__(self, repository: 'Repository', tag: str, release_tags: List[str], push_registries: List['Registry']):
        self.repository = repository
        self.tag:str = tag
        self.push_registries: List[Registry] = push_registries
        self.release_tags = release_tags

        for r in push_registries:
            if r.source:
                raise RuntimeError("Registry %s is marked as 'source'. Can't push there." % (r,))

        if not self.tag in repository.tags:
            raise RuntimeError("Tag %s missing from repository.", tag)

    def __str__(self):
        return str(self.repository) + ":" + self.tag
    __repr__ = __str__


class Repository(object):

    def __init__(self, name: str, registry: 'Registry', tags: List[str], namespace: str = None):
        self.name = name
        self.namespace = namespace
        self.tags: List[str] = tags
        self.registry: Registry = registry

    def __str__(self):
        repository = self.registry.url
        if self.namespace:
            repository += "/" + self.namespace
        repository += "/" + self.name

        return repository
    __repr__ = __str__


class Registry(object):
    def __init__(self, config : dict):
        self.log = logging.getLogger("helper.Registry")
        self.log.setLevel(logging.DEBUG)

        self.name: str = config['name']
        self.url: str = config['url']
        self.catalog: Dict[str] = None
        self.repositories : Dict[Repository] = {}
        self.token = None
        self.client = docker.APIClient(base_url='unix://var/run/docker.sock')

        if 'source' in config:
            self.source = config['source']
        else:
            self.source = False

        if 'target' in config:
            self.target = config['target']
        else:
            self.target = False

        if 'untrusted' in config:
            self.untrusted = config['untrusted']
        else:
            self.untrusted = False

        if 'credentials' in config:
            self.credentials = config['credentials']
        else:
            self.credentials = None

        if 'namespace' in config:
            self.namespace = config['namespace']
        else:
            self.namespace = None

        self._authenticate()
        self._init_raw_client()

    def __str__(self):
        return self.url
    __repr__ = __str__

    def _authenticate(self):
        if not self.credentials:
            return

        self.client.login(username=self.credentials['username'],
                          password=self.credentials['password'],
                          registry=self.url)

    def _init_raw_client(self) -> None:
        """Initialize the raw docker client for interactions with catalog

        This initializes a raw client (based on docker_registry_util.client) that
        handles authentication and interaction with both insecure and secure registries.
        """
        if self.credentials:
            auth = HTTPBasicAuth(self.credentials['username'], self.credentials['password'])
        else:
            auth = None
        base_url = "http://" if self.untrusted else "https://"
        base_url += self.url
        self.raw_client = client.DockerRegistryClient(base_url=base_url, auth=auth)

    def get_repositories(self) -> None:
        """This method fetches all repositories from the registry."""

        self.log.warning("Fetching repositories for %s", self.name)

        catalog = self.raw_client.get_catalog().json()
        self.log.info("Found the following repositories in registry %s:", self.name)
        for repo in catalog['repositories']:
            tags = self.raw_client.get_tags(repo).json()['tags']
            self.log.debug("\t%s with %s tags", repo, len(tags))
            self.repositories[repo] = Repository(name=repo, registry=self, tags=tags)
            self.log.warning(self.repositories[repo])


class ReleaseHelper(object):
    def __init__(self):
        self.log = logging.getLogger("publish.ReleaseHelper")
        self.log.setLevel(logging.DEBUG)

        # command line global arguments
        self.verbose = False
        self.registries : Dict[Registry] = {}
        self.registry_images = {}
        self.image_overrides = {}
        self.source_tag_tpl = ""
        self.release_tags_tpl = {}

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--verbose", dest="verbose", action="store_true")
        parser.add_argument("--release", dest="release", required=True)
        parser.add_argument("--openstack_release", dest="openstack_release", default="ocata")
        parser.add_argument("--build", dest="build_no", required=True)
        parser.add_argument("--dry-run", dest="dry_run", action="store_true")
        parser.add_argument("--config", dest="config", default="publisher.yaml")
        parser.add_argument("--filter", dest="images_filter", default=None)
        args = parser.parse_args()

        self.verbose = args.verbose
        self.release = args.release
        self.build_no = args.build_no
        self.dry_run = args.dry_run
        self.openstack_release = args.openstack_release
        self.images_filter = args.images_filter
        self.config = args.config

    def configure_logging(self):
        pass

    def parse_configuration(self):
        with open(self.config) as fh:
            config = yaml.safe_load(fh)

        for config_reg in config['registries']:
            registry = Registry(config=config_reg)
            self.registries[registry.name] = registry

        self.image_overrides = config['image_overrides']
        self.source_tag_tpl = config['source_tag']
        self.release_tags_tpl = config['release_tags']
        self.distribution = config['distribution']
        self.release_overrides = config['release_overrides']

    def fetch_registry_content(self):
        """Fetch list of repositories from all source registries."""
        for registry_name, registry in self.registries.items():
            if not registry.source:
                continue
            registry.get_repositories()

    def _get_release_override(self, variable) -> Optional[str]:
        return self.release_overrides[variable]

    def get_release(self):
        return self._get_release_override("release")

    def get_build_no(self):
        try:
            return self.release_overrides['build_no']
        except KeyError:
            return self.build_no

    def get_images_from_registry(self, registry: Registry):
        repository: Repository
        images = []
        for repository in registry.repositories.values():
            if self.images_filter and not repository.name in self.images_filter:
                continue

            for tag in repository.tags:
                image = self.get_matching_image_from_repository(repository, tag)
                if image:
                    self.log.warning("Image %s found", image)
                    images += [image]

        return images

    def get_matching_image_from_repository(self, repo: Repository, tag: str) -> Optional[Image]:
        """Returns Image object for the given repository tag

        This method returns Image object for the repository image with a specific tag, applying
        any image-specific overrides defined in publisher.yaml. It can return None if either matching
        wasn't found, or the overrides disabled that image (by setting empty registries).
        """

        # those are default values, and can be candidate on a per-image case.
        distribution = self.distribution
        openstack_release = self.openstack_release
        source_tag_tpl = self.source_tag_tpl
        release_tags_tpl = self.release_tags_tpl
        registries = [r for r in self.registries.values() if not r.source]

        override = {}
        # apply any per-image overrides
        for candidate in self.image_overrides:
            if "image_matcher" in candidate:
                matcher = candidate["image_matcher"]
                if not re.match(matcher, repo.name):
                    continue
            if "tag_matcher" in candidate:
                matcher = candidate["tag_matcher"]
                if not re.match(matcher, tag):
                    continue
            self.log.warning("Overrides for %s:%s found.", repo, tag)
            override = candidate
            break

        if "source_tag" in override:
            source_tag_tpl = override['source_tag']

        if "distribution" in override:
            distribution = override["distribution"]

        if "registries" in override:
            registry_names = override["registries"]
            registries = [r for r in self.registries.values() if r.name in registry_names]

        if "release_tags" in override:
            release_tags_tpl = override["release_tags"]

        # image overrides explicitly disabled processing.
        if not registries:
            self.log.warning("Image %s:%s disabled.", repo, tag)
            return

        source_tag_context = {
            "openstack_release_lhs": openstack_release + "-",
            "openstack_release_rhs": "-" + openstack_release,
            "distribution_lhs": distribution + "-",
            "distribution_rhs": "-" + distribution,
            "release": self.release,
            "build_no": self.build_no,
        }

        release_tag_context = {
            "openstack_release_lhs": openstack_release + "-",
            "openstack_release_rhs": "-" + openstack_release,
            "distribution_lhs": distribution + "-",
            "distribution_rhs": "-" + distribution,
            "release": self.get_release(),
            "build_no": self.get_build_no(),
        }

        def render_tag(template, context):
            template = jinja2.Template(template)
            return template.render(context)

        source_tag = render_tag(source_tag_tpl, source_tag_context)
        release_tags = [render_tag(release_tag, release_tag_context) for release_tag in release_tags_tpl]
        self.log.warning("image %s, tag: %s, source_tag: %s, release_tags: %s",
                         repo.name, tag, source_tag, release_tags)

        # This method is called for every tag in repository, but we want to process only images that
        # match requested release, build_no and openstack version.
        if source_tag == tag:
            image =  Image(repository=repo, tag=source_tag, release_tags=release_tags, push_registries=registries)
            return image
        else:
            return None

    def get_build_images(self) -> List[Image]:
        """Returns a list of all images that are part of the release

        Returns a list of all images from repositories marked as "source" that are part
        of the release. An exception is raised if the same image is in more than one registry.
        """
        images = []
        image_names = []
        conflicting_names = []
        for registry_name, registry in self.registries.items():
            # if the registry is not marked as source, skip it
            if not registry.source:
                continue

            images += self.get_images_from_registry(registry)

        if conflicting_names:
            raise RuntimeError("Images found in multiple 'source' repositories: %s", conflicting_names)

        return images

    def fetch_image(self, image: Image):
        """Fetch image from its registry

        Each `Registry` has its own client so we just access it directly.
        """
        self.log.warning("Fetching image %s", image)
        for line in image.repository.registry.client.pull(str(image.repository), image.tag, stream=True, decode=True):
            self.log.warning(line)

    def tag_image(self, image: Image, target_repository: Repository, tag: str):
        self.log.warning("Tagging %s for %s:%s", image, target_repository, tag)
        image.repository.registry.client.tag(str(image), str(target_repository), tag)

    def publish_image(self, target: Registry, repository: str, tag: str):
        self.log.warning("Pushing %s:%s to %s", repository, tag, target)
        if self.dry_run:
            return
        for line in target.client.push(repository, tag, stream=True, decode=True):
            self.log.warning(line)

    def process_images(self):
        """Get a list of images to publish, tag them and push to registries"""
        source_images = self.get_build_images()
        self.log.warning("Got %s images for publishing. Processing..", len(source_images))

        for image in source_images:
            self.fetch_image(image)

            for target in image.push_registries:
                for tag in image.release_tags:
                    repository = "%s/%s" % (target, image.repository.name)
                    self.tag_image(image, repository, tag)
                    self.publish_image(target, repository, tag)


def main():
    helper = ReleaseHelper()
    helper.parse_arguments()
    helper.configure_logging()
    helper.parse_configuration()
    helper.fetch_registry_content()

    helper.process_images()

if __name__ == "__main__":
    main()

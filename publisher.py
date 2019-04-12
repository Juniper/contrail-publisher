#!/usr/bin/env python3

import logging

from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from docker import APIClient as DockerAPIClient
from docker_registry_util import client
from jinja2 import Template
from re import match as re_match
from requests import HTTPError
from requests.auth import HTTPBasicAuth
from typing import Dict, List, Optional
from urllib3.exceptions import ReadTimeoutError
from yaml import safe_load as yaml_safe_load_file


LOG = logging.getLogger('publisher')
LOG.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - [%(thread)d] %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOG.addHandler(ch)


class ImagePushError(Exception):
    def __init__(self, response=None, message=''):
        self.message = message
        super(ImagePushError).__init__()


class Image(object):
    def __init__(self, repository: 'Repository', tag: str, release_tags: List[str], push_registries: List['Registry']):
        self.repository = repository
        self.tag: str = tag
        self.push_registries: List[Registry] = push_registries
        self.release_tags = release_tags

        for r in push_registries:
            if r.source:
                raise RuntimeError(f'Registry {r} is marked as "source". Cannot push there.')

        if self.tag not in repository.tags:
            raise RuntimeError(f'Tag {tag} missing from repository.')

    def __str__(self):
        return str(self.repository) + ':' + self.tag
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
            repository += '/' + self.namespace
        repository += '/' + self.name

        return repository
    __repr__ = __str__


class Registry(object):
    def __init__(self, config: dict):
        self.log = logging.getLogger('helper.Registry')
        self.log.setLevel(logging.DEBUG)

        self.name: str = config['name']
        self.url: str = config['url']
        self.catalog: Dict[str] = None
        self.repositories: Dict[Repository] = {}
        self.token = None
        self.client = DockerAPIClient(base_url='unix://var/run/docker.sock')

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
        base_url = 'http://' if self.untrusted else 'https://'
        base_url += self.url
        self.raw_client = client.DockerRegistryClient(base_url=base_url, auth=auth)

    def get_repositories(self) -> None:
        """This method fetches all repositories from the registry."""

        self.log.info(f'Fetching repositories for {self.name}')

        catalog = self.raw_client.get_catalog().json()
        self.log.info(f'Found the following repositories in registry {self.name}: {catalog}')
        for repo in catalog['repositories']:
            try:
                tags = self.raw_client.get_tags(repo).json()['tags']
            except HTTPError as e:
                self.log.warning(f'Could not get tags for repository {repo} from registry {self}. {e}')
                continue
            if tags is None:
                tags = []
            self.log.debug(f'\t{repo} with {len(tags)} tags')
            self.repositories[repo] = Repository(name=repo, registry=self, tags=tags)
            self.log.info(self.repositories[repo])


class ReleaseHelper(object):
    def __init__(self):
        # command line global arguments
        self.verbose = False
        self.registries: Dict[str] = {}
        self.registry_images = {}
        self.image_overrides = {}
        self.source_tag_tpl = ""
        self.release_tags_tpl = {}
        self.retry_limit = 5
        self.image_thread_count = 1

    def parse_arguments(self):
        parser = ArgumentParser()
        parser.add_argument('--verbose', dest='verbose', action='store_true')
        parser.add_argument('--release', dest='release', required=True)
        parser.add_argument('--openstack_release', dest='openstack_release', default='ocata')
        parser.add_argument('--build-registry', dest='build_registry', default=None)
        parser.add_argument('--build', dest='build_no', required=True)
        parser.add_argument('--dry-run', dest='dry_run', action='store_true')
        parser.add_argument('--config', dest='config', default='publisher.yaml')
        parser.add_argument('--filter', dest='images_filter', default=None)
        parser.add_argument('--retry-limit', dest='retry_limit', default=self.retry_limit)
        args = parser.parse_args()

        self.verbose = args.verbose
        self.release = args.release
        self.build_no = args.build_no
        self.dry_run = args.dry_run
        self.openstack_release = args.openstack_release
        self.images_filter = args.images_filter
        self.config = args.config
        self.build_registry = args.build_registry
        self.retry_limit = args.retry_limit

    def configure_logging(self):
        self.log = logging.getLogger('publish.ReleaseHelper')
        if self.verbose:
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.INFO)
        logging_handler = logging.StreamHandler()
        logging_handler.setFormatter(logging.Formatter('[%(thread)d] %(levelname)s: %(message)s'))
        self.log.addHandler(logging_handler)

    def parse_configuration(self):
        with open(self.config) as fh:
            config = yaml_safe_load_file(fh)

        for config_reg in config['registries']:
            if config_reg['name'] == 'build' and self.build_registry:
                config_reg['url'] = self.build_registry
            registry = Registry(config=config_reg)
            self.registries[registry.name] = registry

        self.source_tag_tpl = config['source_tag']
        self.release_tags_tpl = config['release_tags']
        self.distribution = config['distribution']

        self.retry_limit = config.get('retry_limit', self.retry_limit)
        self.image_overrides = config.get('image_overrides', {})
        self.release_overrides = config.get('release_overrides', {})
        self.image_thread_count = config.get('image_thread_count', self.image_thread_count)

    def fetch_registry_content(self):
        """Fetch list of repositories from all source registries."""
        for registry_name, registry in self.registries.items():
            if not registry.source:
                continue
            registry.get_repositories()

    def _get_release_override(self, variable) -> Optional[str]:
        return self.release_overrides[variable]

    def get_release(self):
        try:
            return self.release_overrides['release']
        except KeyError:
            return self.release

    def get_build_no(self):
        try:
            return self.release_overrides['build_no']
        except KeyError:
            return self.build_no

    def get_images_from_registry(self, registry: Registry):
        repository: Repository
        images = []
        for repository in registry.repositories.values():
            if self.images_filter and repository.name not in self.images_filter:
                continue

            for tag in repository.tags:
                image = self.get_matching_image_from_repository(repository, tag)
                if image:
                    self.log.info(f'Image {image} found')
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
            if 'image_matcher' in candidate:
                matcher = candidate['image_matcher']
                if not re_match(matcher, repo.name):
                    continue
            if 'tag_matcher' in candidate:
                matcher = candidate['tag_matcher']
                if not re_match(matcher, tag):
                    continue
            # self.log.warning(f'Overrides for {repo}:{tag} found.')
            override = candidate
            break

        if 'source_tag' in override:
            source_tag_tpl = override['source_tag']

        if 'distribution' in override:
            distribution = override['distribution']

        if 'registries' in override:
            registry_names = override['registries']
            registries = [r for r in self.registries.values() if r.name in registry_names]

        if 'release_tags' in override:
            release_tags_tpl = override['release_tags']

        # image overrides explicitly disabled processing.
        if not registries:
            self.log.info(f'Image {repo}:{tag} disabled.')
            return

        source_tag_context = {
            'openstack_release_lhs': openstack_release + '-',
            'openstack_release_rhs': '-' + openstack_release,
            'distribution_lhs': distribution + '-',
            'distribution_rhs': '-' + distribution,
            'release': self.release,
            'build_no': self.build_no,
        }

        release_tag_context = {
            'openstack_release_lhs': openstack_release + '-',
            'openstack_release_rhs': '-' + openstack_release,
            'distribution_lhs': distribution + '-',
            'distribution_rhs': '-' + distribution,
            'release': self.get_release(),
            'build_no': self.get_build_no(),
        }

        def render_tag(template, context):
            template = Template(template)
            return template.render(context)

        source_tag = render_tag(source_tag_tpl, source_tag_context)
        release_tags = [render_tag(release_tag, release_tag_context) for release_tag in release_tags_tpl]

        # This method is called for every tag in repository, but we want to process only images that
        # match requested release, build_no and openstack version.
        if source_tag == tag:
            image = Image(repository=repo, tag=source_tag, release_tags=release_tags, push_registries=registries)
            return image
        else:
            return None

    def get_build_images(self) -> List[Image]:
        """Returns a list of all images that are part of the release

        Returns a list of all images from repositories marked as "source" that are part
        of the release. An exception is raised if the same image is in more than one registry.
        """
        images = []
        conflicting_names = []
        for registry_name, registry in self.registries.items():
            # if the registry is not marked as source, skip it
            if not registry.source:
                continue

            images += self.get_images_from_registry(registry)

        if conflicting_names:
            raise RuntimeError(f'Images found in multiple source repositories: {conflicting_names}')

        return images

    def fetch_image(self, image: Image):
        """Fetch image from its registry

        Each `Registry` has its own client so we just access it directly.
        """
        self.log.info(f'Fetching image {image}')
        for line in image.repository.registry.client.pull(str(image.repository), image.tag, stream=True, decode=True):
            self.log.debug(line)

    def tag_image(self, image: Image, target_repository: Repository, tag: str):
        self.log.info(f'Tagging {image} for {target_repository}:{tag}')
        image.repository.registry.client.tag(str(image), str(target_repository), tag)

    def publish_image(self, target: Registry, repository: str, tag: str):
        if self.dry_run:
            return
        try:
            for line in target.client.push(repository, tag, stream=True, decode=True):
                if 'error' in line.keys():
                    message = line['errorDetail']['message']
                    raise ImagePushError(message=message)
                self.log.debug(line)
        except ReadTimeoutError as e:
            raise ImagePushError(message=str(e))
        return

    def _process_image(self, image):
        self.fetch_image(image)

        for target in image.push_registries:
            for tag in image.release_tags:
                repository = f'{target}/{image.repository.name}'
                self.tag_image(image, repository, tag)
                retry_count = 1
                while retry_count <= self.retry_limit:
                    self.log.info(f'Pushing {repository}:{tag} to'
                                  f' {target} ({retry_count}/{self.retry_limit})')
                    try:
                        self.publish_image(target, repository, tag)
                        break
                    except ImagePushError as e:
                        self.log.error(f'Failed {retry_count}/{self.retry_limit} attempt to push image {image}.'
                                       f' Error: {e.message}')
                        retry_count = retry_count + 1
                else:
                    return False
        return True

    def process_all_images(self):
        """Get a list of images to publish, tag them and push to registries"""
        source_images = self.get_build_images()
        if self.image_thread_count == 0 or len(source_images) < self.image_thread_count:
            thread_count = len(source_images)
        else:
            thread_count = self.image_thread_count
        self.log.info(f'Got {len(source_images)} images for publishing.'
                      f' Processing in {thread_count} threads.')

        thread_results = []
        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            # Maping dict with keys as futures and images as values
            future_to_image = {executor.submit(self._process_image, image): image for image in source_images}
            for future in as_completed(future_to_image):
                image = future_to_image[future]
                try:
                    thread_results.append(future.result())
                except Exception as e:
                    self.log.error(f'Error processing {image}. {e}')
                    thread_results.append(False)
                else:
                    self.log.info(f'Image {image} processing finished. Result {thread_results[-1]}.')

        return all(thread_results)


def main():
    helper = ReleaseHelper()
    helper.parse_arguments()
    helper.configure_logging()
    helper.parse_configuration()
    helper.fetch_registry_content()

    push_succeeded = helper.process_all_images()
    if not push_succeeded:
        helper.log.error('There were errors while pushing images. Aborting...')
        return 1
    helper.log.info('Publishing completed successfully')
    return 0


if __name__ == '__main__':
    exit(main())

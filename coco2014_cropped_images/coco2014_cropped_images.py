"""coco2014_cropped_images dataset."""
import os
import tensorflow_datasets as tfds
import pathlib
from pathlib import Path

project_root_path = Path(__file).parent.parent.resolve()
coco_bbox_images_path = os.path.join(project_root_path, 'dataset', 'coco', 'bbox_images')
train_path = os.path.join(coco_path, 'train2014')
validation_path = os.path.join(coco_path, 'validation2014')
test_path = os.path.join(coco_path, 'test2014')

# TODO(coco2014_cropped_images): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(coco2014_cropped_images): BibTeX citation
_CITATION = """
"""

# Get all label names from label.txt
def get_label_mapping_list():
  label_mapping_file = open(os.path.join(project_root_path, 'coco2014_labels.txt'), 'r')
  label_mapping_list = label_mapping_file.readlines()
  label_mapping_file.close()
  label_mapping_list = list(map(lambda x : x.replace('\n', ''), label_mapping_list))
  return label_mapping_list[0:80]

# Get label name and unique id for each extracted image
def get_label_and_id_from_file_name(path):
  # path is pathlib.WindowsPath
  path_parts = path.parts
  image_name = path_parts[len(path_parts) - 1]
  path_str = str(image_name)
  path_split = path_str.split('_')
  label = path_split[0]
  id = path_split[len(path_split) - 1].split('.')
  id = id[0]
  return (id, label)

class Coco2014CroppedImages(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for coco2014_cropped_images dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(coco2014_cropped_images): Specifies the tfds.core.DatasetInfo object
    label_mapping_list = get_label_mapping_list()
    print('label_mapping_list ', label_mapping_list)
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label': tfds.features.ClassLabel(names=label_mapping_list),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(coco2014_cropped_images): Downloads the data and defines the splits
    
    # TODO(coco2014_cropped_images): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(Path(train_path)),
        'validation': self._generate_examples(Path(validation_path)),
        'test': self._generate_examples(Path(test_path))
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(coco2014_cropped_images): Yields (key, example) tuples from the dataset
    for f in path.glob('*.jpeg'):
      id, label = get_label_and_id_from_file_name(f)
      if not id.isdecimal():
        raise Exception('id is wrong.')
      yield id, {
          'image': f,
          'label': label,
      }

import json
from pathlib import Path
from cogworks_data.language import get_data_path


class COCOData:
    def __init__(self, valid_image_ids):
        file_path = get_data_path("captions_train2014.json")

        with Path(file_path).open() as file:
            coco_data = json.load(file)

        valid_image_ids = set(valid_image_ids)

        self.image_ids = []
        self.caption_ids = []
        self.image_to_captions = {}
        self.caption_to_image = {}
        self.caption_text = {}

        for image in coco_data["images"]:
            image_id = image["id"]
            if image_id in valid_image_ids:
                self.image_ids.append(image_id)
                self.image_to_captions[image_id] = []

        for annotation in coco_data["annotations"]:
            image_id = annotation["image_id"]
            if image_id not in self.image_to_captions:
                continue
            caption_id = annotation["id"]

            self.caption_ids.append(caption_id)
            self.image_to_captions[image_id].append(caption_id)
            self.caption_to_image[caption_id] = image_id
            self.caption_text[caption_id] = annotation["caption"]
from ssd.dataset import COCODataset

ds = COCODataset("val")
# print(ds[0])
# import json
# from pathlib import Path

# content = Path("datasets/COCO/annotations/instances_val2017.json").read_text()
# content = json.loads(content)

# # print([x["category_id"] for x in content["annotations"]])
# print(len(content["categories"]))
print(ds.dataset.ids)

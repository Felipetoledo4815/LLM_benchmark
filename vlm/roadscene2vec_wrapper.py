from typing import List, Tuple
from time import time
import cv2
import matplotlib.pyplot as plt
from vlm.vlm_interface import VLMInterface
from external_code.roadscene2vec.roadscene2vec.util.config_parser import configuration
from external_code.roadscene2vec.roadscene2vec.util.visualizer import draw_bbox
from external_code.roadscene2vec.roadscene2vec.scene_graph.extraction.image_extractor import RealExtractor
from external_code.roadscene2vec.roadscene2vec.scene_graph.scene_graph import SceneGraph

class RoadScene2Vec(VLMInterface):
    #TODO: This should not be a VLMInterface, it should be a SGGInterface
    def __init__(self, config_file: str) -> None:
        self.config_file = config_file
        self.config = configuration(config_file, from_function = True)
        self.extractor = RealExtractor(self.config)
        self.mapping = {
            "toLeftOf": "to_left_of",
            "toRightOf": "to_right_of",
            "inFrontOf": "in_front_of"
        }

    def inference(self, prompt: str, images: List[str], **kwargs) -> Tuple[str, float]:
        assert len(images) == 1, "RoadScene2Vec only supports one image at a time."
        img = cv2.imread(str(images[0]), cv2.IMREAD_COLOR) 

        start = time()
        bbox = self.extractor.get_bounding_boxes(img)
        bev = self.extractor.bev
        scenegraph = SceneGraph(self.extractor.relation_extractor, bounding_boxes=bbox, bev=bev, 
                                coco_class_names=self.extractor.coco_class_names, platform=self.extractor.dataset_type)
        # bbox_img = draw_bbox(self.extractor, img)
        # plt.imshow(cv2.cvtColor(bbox_img, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.savefig("output.png")
        response_str = self.nxsg2str(scenegraph.g)
        end = time()

        return response_str, end - start

    def nxsg2str(self, nx_sg):
        triplets = []
        triplets_final = []
        for source, target, data in nx_sg.edges(data=True):
            if target.label == "ego_car" and data['label'] != "isIn":
                if data['label'] in self.mapping:
                    triplets.append((source.name, self.mapping[data['label']], 'ego'))
                else:
                    triplets.append((source.name, data['label'], 'ego'))
        for t in triplets:
            triplets_final.append((t[0].split("_")[0], t[1], t[2]))
        return str(triplets_final)

    def parse_prompt(self, prompt: str, images: List[str], **kwargs) -> None:
        raise NotImplementedError("RoadScene2Vec does not support prompt parsing.")

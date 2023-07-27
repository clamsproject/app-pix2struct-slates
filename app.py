"""
Pix2structSlates runs the Pix2struct Doc-VQA model on timeframes from a MMIF containing a VideoDocument.
It specifically runs the model on the slate of the video with queries about the slate text.
These queries are:
    - What is the title of the program
    - What date was it recorded
    - What date did it air
    - What is the total runtime of the program
"""
import argparse
import logging
from typing import Union, List, Dict, Tuple, Iterable

# mostly likely you'll need these modules/classes
from clams import ClamsApp, Restifier
from mmif import Mmif, View, Annotation, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh

import torch
from transformers import Pix2StructForConditionalGeneration as psg
from transformers import Pix2StructProcessor as psp


class Pix2structSlates(ClamsApp):

    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = psg.from_pretrained("google/pix2struct-docvqa-base").to(self.device)
        self.processor = psp.from_pretrained("google/pix2struct-docvqa-base")

    def _appmetadata(self):
        # see https://sdk.clams.ai/autodoc/clams.app.html#clams.app.ClamsApp._load_appmetadata
        # Also check out ``metadata.py`` in this directory.
        # When using the ``metadata.py`` leave this do-nothing "pass" method here.
        pass

    def generate(self, img, questions):
        """
        Generate answers for a list of questions using the model
        :param img:
        :param questions:
        :return:
        """
        inputs = self.processor(images=[img for _ in range(len(questions))],
                                text=questions, return_tensors="pt").to(self.device)
        predictions = self.model.generate(**inputs, max_new_tokens=256)
        return zip(questions, self.processor.batch_decode(predictions, skip_special_tokens=True))

    def _annotate(self, mmif: Union[str, dict, Mmif], **parameters) -> Mmif:
        video_doc: Document = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
        input_view: View = mmif.get_views_for_document(video_doc.properties.id)[0]

        config = self.get_configuration(**parameters)
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(
            AnnotationTypes.Relation,
            document=video_doc.id,
        )

        query_to_label = {
            "What is the title of the program": "title",
            "What date was is recorded": "rec_date",
            "What date did it air": "air_date",
            "What is the total runtime of the program": "runtime"
        }

        queries = list(query_to_label.keys())

        for timeframe in input_view.get_annotations(AnnotationTypes.TimeFrame, frameType="slate"):
            self.logger.debug(timeframe.properties)
            # get image from time frame
            image = vdh.extract_mid_frame(mmif, timeframe, as_PIL=True)
            completions = self.generate(image, queries)

            for query, answer in completions:
                print(f"query: {query} answer: {answer}")
            # add question answer pairs as properties to timeframe
                text_document = new_view.new_textdocument(answer)
                text_document.add_property("query", query)
                text_document.add_property("label", query_to_label[query])
                align_annotation = new_view.new_annotation(AnnotationTypes.Alignment)
                align_annotation.add_property("source", timeframe.id)
                align_annotation.add_property("target", text_document.id)
            pass

        return mmif


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    # add more arguments as needed
    # parser.add_argument(more_arg...)

    parsed_args = parser.parse_args()

    # create the app instance
    app = Pix2structSlates()

    http_app = Restifier(app, port=int(parsed_args.port))
    # for running the application in production mode
    if parsed_args.production:
        http_app.serve_production()
    # development mode
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()

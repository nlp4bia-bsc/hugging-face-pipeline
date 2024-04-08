# Loading script for the MEDDOPLACE NER dataset (LOC version).
import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{miranda2020named,
  title={Named entity recognition, concept normalization and clinical coding: Overview of the cantemist track for cancer text mining in spanish, corpus, guidelines, methods and results},
  author={Miranda-Escalada, A and Farr{\'e}, E and Krallinger, M},
  booktitle={Proceedings of the Iberian Languages Evaluation Forum (IberLEF 2020), CEUR Workshop Proceedings},
  year={2020}
}"""


_DESCRIPTION = """\
MEDDOPLACE dataset contains samples with annotated procedures in clinic documents.
NLP for Biomedical Information Analysis.
"""

# MODIFY THIS
#_BASE_URL = f"https://huggingface.co/datasets/<your-username>/<dataset-name>/raw/main/"
# If training locally, you can just specify the path where the CoNLLs are:
_BASE_URL = "./"
_URLS = {
    "train": _BASE_URL + "train.conll",
    "validation": _BASE_URL + "validation.conll",
    "test": _BASE_URL + "test.conll",
}


class MeddoplaceConfig(datasets.BuilderConfig):
    """BuilderConfig for MEDDOPLACE."""

    def __init__(self, **kwargs):
        """BuilderConfig for MEDDOPLACE.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(MeddoplaceConfig, self).__init__(**kwargs)


class Meddoplace(datasets.GeneratorBasedBuilder):
    """Meddoplace NER dataset"""

    BUILDER_CONFIGS = [
        MeddoplaceConfig(
            name="Meddoplace",
            version=datasets.Version("1.0.0"),
            description="Meddoplace dataset",
        ),
    ]

    def _get_bio_tags(self):
        CLASS_NAMES = [
            'GPE',
            'GEO', 
            'FAC', 
            'COMUNIDAD', 
            'IDIOMA', 
            'DEPARTAMENTO', 
            'TRANSPORTE'
        ]
        bio_tags = ["O"]
        for class_name in CLASS_NAMES:
            bio_tags.append("B-" + class_name)
            bio_tags.append("I-" + class_name)
        return bio_tags

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=self._get_bio_tags()
                        )
                    ),
                }
            ),
            homepage="https://temu.bsc.es/meddoplace/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = dl_manager.download_and_extract(_URLS)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                                    "filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                                    "filepath": downloaded_files["validation"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                                    "filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                # End of the sample
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "ner_tags": ner_tags,
                        }
                        # Init new sample
                        guid += 1
                        tokens = []
                        ner_tags = []
                # While not end of sample, append token and ner tag
                else:
                    # Meddoplace .conll files are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[3].rstrip())
                    ner_tags.append(splits[0])
            # Last example, if any
            if tokens:
                yield guid, {
                    "id": str(guid),
                    "tokens": tokens,
                    "ner_tags": ner_tags,
                }

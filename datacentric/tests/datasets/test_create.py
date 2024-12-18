import pytest
from google.cloud import storage
from datacentric import config
from datacentric import datasets
from datacentric.evals import types
from google.cloud import aiplatform
import pandas as pd
from google.cloud.aiplatform import schema


def test_function_export():
    assert datasets.do_one() == 1

@pytest.fixture
def cfg():
    return config.new(config.Mode.DEV)

def test_bucket_exists(cfg: config.Config):
    client = storage.Client(project=cfg.project_id)
    bucket = client.lookup_bucket(cfg.bucket_name)
    if bucket is None:
        raise Exception(f"Bucket {cfg.bucket_name} does not exist.")

def test_schema_exists(cfg: config.Config):
    schema_blob = 'schemas/genai.yaml'
    client = storage.Client(project=cfg.project_id)
    bucket = client.bucket(cfg.bucket_name)
    blob = bucket.blob(schema_blob)
    if not blob.exists():
        raise Exception(f"Schema file {schema_blob} does not exist.")
    

def test_create_new_dataset(cfg: config.Config):
    display_name = 'datacentric-expanded-123'
    gcs_source = "gs://ruinard_datacentric/data/question_answering.csv"
    # metadata_schema_uri = 'gs://google-cloud-aiplatform/schema/dataset/metadata/text_1.0.0.yaml'
    # import_schema_uri = 'gs://ruinard_datacentric/schemas/genai.yaml'
    datasets.new(cfg.project_id, display_name=display_name, gcs_source=gcs_source)

def test_create_new_text_dataset(cfg: config.Config):
    display_name = 'datacentric-text'
    gcs_source = "gs://ruinard_datacentric/data/question_answering.jsonl"
    # metadata_schema_uri = 'gs://google-cloud-aiplatform/schema/dataset/metadata/text_1.0.0.yaml'
    # import_schema_uri = 'gs://ruinard_datacentric/schemas/custom/question_answering.yaml'
    import_schema_uri = schema.dataset.ioformat.text.single_label_classification

    datasets.new_text_ds(cfg.project_id, display_name=display_name, gcs_source=gcs_source, import_schema_uri=import_schema_uri)

def test_create_new_text_dataset_with_question_only(cfg: config.Config):
    display_name = 'datacentric-text-question-only'
    gcs_source = "gs://ruinard_datacentric/data/questions_only.jsonl"
    # metadata_schema_uri = 'gs://google-cloud-aiplatform/schema/dataset/metadata/text_1.0.0.yaml'
    import_schema_uri = 'gs://ruinard_datacentric/schemas/ioformat/custom/question_only.yaml'
    

    datasets.new_text_ds(cfg.project_id, display_name=display_name, gcs_source=gcs_source, import_schema_uri=import_schema_uri)
    
    
def test_export_dataset(cfg: config.Config):

    # ds = ads.TabularDataset.list(filter=f'labels.eval_type="{types.EvalTypes.QUESTION_ANSWERING.value}"', order_by='update_time')
    # ds = ads.TabularDataset.list(filter=f'displayName="datacentric-expanded-123"', order_by='update_time')

    # assert len(ds) == 2

    ds = aiplatform.TabularDataset.list(filter=f'labels.eval_type="{types.EvalTypes.QUESTION_ANSWERING.value}" AND displayName="datacentric-expanded-123"', order_by='update_time')
    assert len(ds) == 3


def test_load_data(cfg: config.Config):

    # ds = aiplatform.TabularDataset.list(filter=f'labels.eval_type="{types.EvalTypes.QUESTION_ANSWERING.value}"', order_by='update_time')
    # ds = aiplatform.TabularDataset.list(filter=f'displayName="datacentric-expanded-123"', order_by='update_time')

    # assert len(ds) == 2

    ds = aiplatform.TabularDataset.list(filter=f'labels.eval_type="{types.EvalTypes.QUESTION_ANSWERING.value}" AND displayName="datacentric-expanded-123"', order_by='update_time')
    ds_of_interest = ds[0]
    input_config = ds_of_interest.to_dict()['metadata']['inputConfig']
    data = pd.concat([pd.read_csv(gcs_path) for gcs_path in input_config['gcsSource']['uri']])
    assert len(data) == 5

    
def test_obtain_artifact(cfg: config.Config):
    aiplatform.init(project=cfg.project_id, location=cfg.location)

    artifacts = aiplatform.Artifact.list()
    print('artifacts')
    artifact = artifacts[0]
    dataset_resource_name = artifact.to_dict()['metadata']['resourceName']
    dataset = aiplatform.TabularDataset(dataset_resource_name)
    input_config = dataset.to_dict()['metadata']['inputConfig']
    data = pd.concat([pd.read_csv(gcs_path) for gcs_path in input_config['gcsSource']['uri']])
    assert len(data) == 5

def test_load_data_for_text_dataset(cfg: config.Config):

    # ds = aiplatform.TabularDataset.list(filter=f'labels.eval_type="{types.EvalTypes.QUESTION_ANSWERING.value}"', order_by='update_time')
    # ds = aiplatform.TabularDataset.list(filter=f'displayName="datacentric-expanded-123"', order_by='update_time')

    # assert len(ds) == 2

    ds = aiplatform.TextDataset.list(filter=f'labels.eval_type="{types.EvalTypes.QUESTION_ANSWERING.value}" AND displayName="datacentric-text-question-only"', order_by='update_time')
    ds_of_interest = ds[0]
    print(ds_of_interest.to_dict())
    input_config = ds_of_interest.to_dict()['metadata']['inputConfig']
    # data = pd.concat([pd.read_csv(gcs_path) for gcs_path in input_config['gcsSource']['uri']])
    # assert len(data) == 5
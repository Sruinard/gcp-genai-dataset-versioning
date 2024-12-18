from google.cloud import storage
from google.cloud import aiplatform
from datacentric.evals import types
from google.cloud.aiplatform import schema
schema.dataset.ioformat.text.multi_label_classification


def do_one():
    return 1

def upload_to_gcs(local_file_path: str, bucket_name: str, destination_blob_name: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"File {local_file_path} uploaded to {destination_blob_name} in {bucket_name} bucket.")

def new(
    project: str,
    display_name: str,
    gcs_source: str,
    # metadata_schema_uri: str,
    # import_schema_uri: str,
    location: str = "us-central1",
):
    ds = aiplatform.TabularDataset.create(
        display_name=display_name,
        gcs_source=gcs_source,
        project=project,
        location=location,
        labels={
            "eval_type": types.EvalTypes.QUESTION_ANSWERING.value
        }

        # metadata_schema_uri=metadata_schema_uri,
        # import_schema_uri=[import_schema_uri],
    )

def new_text_ds(
    project: str,
    display_name: str,
    gcs_source: str,
    # metadata_schema_uri: str,
    import_schema_uri: str,
    location: str = "us-central1",
):
    ds = aiplatform.TextDataset.create(
        display_name=display_name,
        gcs_source=gcs_source,
        import_schema_uri=import_schema_uri,
        project=project,
        location=location,
        labels={
            "eval_type": types.EvalTypes.QUESTION_ANSWERING.value
        }

        # metadata_schema_uri=metadata_schema_uri,
        # import_schema_uri=[import_schema_uri],
    )
    print('done')



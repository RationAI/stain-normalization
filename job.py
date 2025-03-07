from kube_jobs import Storage, submit_job


submit_job(
    job_name="stain-normalization-train",
    username="xlopatka",
    cpu=10,
    memory="128Gi",
    gpu="A40",
    shm="16384Mi",
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/libraries/staining.git",
        "cd staining",
        "git checkout feature/ml-stain-normalization",
        "pdm sync --skip=post_install",
        "pdm train",
    ],
    storage=Storage(mou=True),
)

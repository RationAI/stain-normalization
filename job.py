from kube_jobs import Storage, submit_job


submit_job(
    job_name="some name",
    username="your name",
    cpu=10,
    memory="128Gi",
    gpu="A40",
    script=[
        "git clone https://gitlab.ics.muni.cz/rationai/digital-pathology/... workdir",
        "cd workdir",
        "git checkout develop",
        "pdm sync --skip=post_install",
        "pdm fit model/backbone=resnet18",
    ],
    storage=Storage(mou=True),
)

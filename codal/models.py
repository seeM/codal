from django.db import models


class Organization(models.Model):
    name = models.TextField(unique=True)


class Repo(models.Model):
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    name = models.TextField()
    zip_path = models.TextField(unique=True, null=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["organization", "name"], name="unique_repo_org_name"
            )
        ]


class Document(models.Model):
    repo = models.ForeignKey(Repo, on_delete=models.CASCADE)
    path = models.TextField()
    text = models.TextField()

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["repo", "path"], name="unique_document_repo_path"
            )
        ]


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    start = models.IntegerField()
    end = models.IntegerField()
    embedding = models.OneToOneField("Embedding", on_delete=models.CASCADE)

    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["document", "start"], name="unique_chunk_document_start"
            ),
            models.UniqueConstraint(
                fields=["document", "end"], name="unique_chunk_document_end"
            ),
        ]


class Embedding(models.Model):
    path = models.TextField(unique=True)
    hash = models.TextField(unique=True)

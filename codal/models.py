from django.db import models


class Organization(models.Model):
    name = models.TextField()


class Repo(models.Model):
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE)
    name = models.TextField()


class Document(models.Model):
    repo = models.ForeignKey(Repo, on_delete=models.CASCADE)
    path = models.TextField()
    text = models.TextField()


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE)
    start = models.IntegerField()
    end = models.IntegerField()
    embedding = models.OneToOneField("Embedding", on_delete=models.CASCADE)


class Embedding(models.Model):
    path = models.TextField()
    hash = models.TextField()
    chunk = models.OneToOneField("Chunk", on_delete=models.CASCADE)

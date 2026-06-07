from pathlib import Path

from pydantic import BaseModel


class SupplierModelInfo(BaseModel):
    @property
    def license(self):
        raise Exception("Abstract method")

    @license.setter
    def license(self, value):
        raise Exception("Abstract method")

    @property
    def license_url(self):
        if self.license:
            return self.license.license_url
        raise ValueError("undefined license")

    @property
    def supplier(self):
        raise Exception("Abstract method")

    @property
    def template(self):
        raise Exception("Abstract method")

    def save(self, file_path: str | Path):
        raise Exception("Abstract method")

from typing import Optional, Any

from pydantic import BaseModel

from core.model.models_constants import COMMON_LICENSE
from core.model.storage_helpers.SupplierFileInfo import Supplier
from core.model.SupplierModelInfo import SupplierModelInfo


class ModelLicense(BaseModel):
    supplier: Optional[Supplier] = None
    license_name: str = None
    license_url: Optional[str] = None
    license_text: str = None

    @staticmethod
    def from_modelfile_license(
        license_text: str, supplier_model_file_info: None | SupplierModelInfo = None
    ) -> Any:
        if license_text:
            if license_text.startswith('"""') and license_text.endswith('"""'):
                license_text = license_text[3:-3]
                model_license: ModelLicense = ModelLicense(
                    license_name=ModelLicense.common_license_from_text(
                        license_text=license_text
                    ),
                    license_text=license_text,
                )
                if supplier_model_file_info:
                    model_license.supplier = supplier_model_file_info.supplier
                    model_license.license_url = supplier_model_file_info.license_url
                return model_license
        return None

    @staticmethod
    def from_content(content: str):
        return ModelLicense(
            license_name=ModelLicense.common_license_from_text(content),
            license_text=content,
        )

    @staticmethod
    def common_license_from_text(license_text: str, license_name: str = None) -> str:
        if license_name:
            if license_name != DEFAULT_MODEL_LICENSE_NAME:
                return license_name
        for lic_name, lic_content_match in COMMON_LICENSE:
            if lic_content_match in license_text:
                return lic_name
        if license_text:
            for line in license_text.splitlines()[:5]:
                l_content = line.replace("#", " ").strip()
                if len(l_content) > 3:
                    return l_content
        return DEFAULT_MODEL_LICENSE_NAME

    @property
    def common_license(self):
        if self.license_name:
            lic_name = ModelLicense.common_license_from_text(
                license_text=self.license_text, license_name=self.license_name
            )
            if lic_name != DEFAULT_MODEL_LICENSE_NAME:
                self.license_name = lic_name
            return lic_name
        return DEFAULT_MODEL_LICENSE_NAME

    @property
    def license_link(self, model_file_info=None):
        if self.license_url:
            return self.license_url
        elif self.supplier and model_file_info:
            if self.supplier is not None and model_file_info.license:
                self.license_url = model_file_info.license_url
                return self.license_url
        return None


DEFAULT_MODEL_LICENSE_NAME = "other"

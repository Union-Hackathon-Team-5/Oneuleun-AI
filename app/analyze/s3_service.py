import logging
import os
import uuid
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Uploader:
    """
    Lightweight helper for uploading media into S3 with configurable prefixes.
    Attempts to apply the ACL configured via S3_OBJECT_ACL (default public-read) but
    will fall back to bucket defaults when ACLs are disallowed.
    """

    def __init__(self):
        access_key = os.getenv("AWS_ACCESS_KEY_ID")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION")
        bucket = os.getenv("S3_BUCKET_NAME")
        public_base = os.getenv("S3_PUBLIC_BASE")
        object_acl_env = os.getenv("S3_OBJECT_ACL", "public-read")

        if not all([access_key, secret_key, region, bucket, public_base]):
            raise ValueError(
                "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET_NAME, S3_PUBLIC_BASE 환경 변수 설정이 필요합니다."
            )

        self.bucket = bucket
        self.region = region
        self.public_base = public_base.rstrip("/")
        normalized_acl = (object_acl_env or "").strip().lower()
        if normalized_acl in {"", "none", "disabled", "off"}:
            normalized_acl = ""
        self.object_acl = normalized_acl or None
        self.client = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )

    def upload_media(
        self,
        *,
        content: bytes,
        session_id: str,
        filename: Optional[str],
        content_type: Optional[str],
        prefix: str,
    ) -> str:
        if not content:
            raise ValueError("업로드할 데이터가 비어있습니다.")

        original = Path(filename or "file.bin")
        ext = original.suffix or ".bin"
        normalized_prefix = prefix.strip("/")
        bucket_prefix = f"{self.bucket}/"
        if normalized_prefix.startswith(bucket_prefix):
            normalized_prefix = normalized_prefix[len(bucket_prefix):]

        key = f"{normalized_prefix.rstrip('/')}/{session_id}/{uuid.uuid4().hex}{ext}"

        extra_args = {}
        if content_type:
            extra_args["ContentType"] = content_type

        def put_object(include_acl: bool) -> None:
            call_args = dict(extra_args)
            if include_acl and self.object_acl:
                call_args["ACL"] = self.object_acl
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=content,
                **call_args,
            )

        try:
            put_object(include_acl=bool(self.object_acl))
        except ClientError as error:
            error_code = error.response.get("Error", {}).get("Code")
            if (
                self.object_acl
                and error_code == "AccessControlListNotSupported"
            ):
                logger.warning(
                    "Bucket %s does not support ACL '%s'; retrying without ACL",
                    self.bucket,
                    self.object_acl,
                )
                put_object(include_acl=False)
            else:
                raise

        if self.object_acl:
            logger.info(
                "Uploaded media to s3://%s/%s with ACL %s",
                self.bucket,
                key,
                self.object_acl,
            )
        else:
            logger.info("Uploaded media to s3://%s/%s", self.bucket, key)
        return key

    def upload_audio(
        self,
        *,
        content: bytes,
        session_id: str,
        filename: Optional[str],
        content_type: Optional[str],
        prefix: str = "oneuld/audio",
    ) -> str:
        return self.upload_media(
            content=content,
            session_id=session_id,
            filename=filename,
            content_type=content_type,
            prefix=prefix,
        )

    def upload_image(
        self,
        *,
        content: bytes,
        session_id: str,
        filename: Optional[str],
        content_type: Optional[str],
        prefix: str = "oneuld/image",
    ) -> str:
        return self.upload_media(
            content=content,
            session_id=session_id,
            filename=filename,
            content_type=content_type,
            prefix=prefix,
        )

    def build_public_url(self, key: str) -> str:
        url = f"{self.public_base}/{key}".rstrip(".")
        return url

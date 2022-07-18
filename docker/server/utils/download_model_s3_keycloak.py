import boto3
import argparse

import botocore
from botocore.client import Config
from keycloak import KeycloakOpenID


def get_args():
    parser = argparse.ArgumentParser(
        description="MinIO PoC demonstrating the authentication using Keycloak's OICD and loading data of different users into different MinIO buckets."
    )
    parser.add_argument(
        "--username",
        type=str,
        help="Username of the first user.",
    )
    parser.add_argument(
        "--password",
        type=str,
        help="Password of the first user.",
    )
    parser.add_argument(
        "--object-path",
        type=str,
        help="Path of the object in the form of <bucket>/<object_key>",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to store the object. The directory has to exist.",
    )
    parser.add_argument(
        "--minio-endpoint-url",
        type=str,
        help="Endpoint URL of the MinIO object sotre.",
    )
    parser.add_argument(
        "--keycloak-endpoint-url",
        type=str,
        default="http://localhost:8080/auth/",
        help="Auth endpoint URL of Keycloak.",
    )
    parser.add_argument(
        "--keycloak-client-id",
        type=str,
        default="account",
        help="Keycloak's auth endpoint URL.",
    )
    parser.add_argument(
        "--keycloak-realm-name",
        type=str,
        default="Agri-Gaia",
        help="Keycloak realm name.",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="Don't use ssl.",
    )
    args = parser.parse_args()
    return args


ARGS = get_args()


class OpenID:
    def __init__(
        self,
        username,
        password,
        server_url,
        client_id,
        realm_name,
        client_secret_key=None,
    ):
        self.server_url = server_url
        self.client_id = client_id
        self.realm_name = realm_name
        self.client_secret_key = client_secret_key

        self.username = username
        self.password = password

        self.oid = self._create_oid_client()

    def _create_oid_client(self):
        oid = KeycloakOpenID(
            server_url=self.server_url,
            client_id=self.client_id,
            realm_name=self.realm_name,
            client_secret_key=self.client_secret_key,
        )
        print(
            f'[OpenID::_create_oid_client] Created OpenID client ("{self.server_url}") for user "{self.username}" in realm "{self.realm_name}".'
        )
        return oid

    def get_tokens(self):
        token_response = self.oid.token(self.username, self.password)
        return token_response["access_token"], token_response["refresh_token"]


class MinioS3:
    def __init__(self, endpoint_url, access_token, refresh_token, use_ssl=True):
        self.endpoint_url = endpoint_url
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.use_ssl = use_ssl

        self.sts = self._create_sts_client()
        self.sts_role = self._sts_assume_role()
        self.sts_creds = self.sts_role["Credentials"]

        self.s3 = self._create_s3_resource()

    def _create_sts_client(self):
        print(
            f'[MinioS3::_create_sts_client] Creating STS client for endpoint "{self.endpoint_url}".'
        )
        return boto3.client(
            "sts",
            use_ssl=self.use_ssl,
            endpoint_url=self.endpoint_url,
        )

    def _sts_assume_role(
        self, role_arn="arn:aws:iam::123456789", role_session_name="minios3"
    ):
        print(
            f'[MinioS3::_sts_assume_role] Trying to assume role "{role_arn}" with WebIdentity through the Security Token Service (STS)...'
        )
        sts_role = self.sts.assume_role_with_web_identity(
            RoleArn=role_arn,
            RoleSessionName=role_session_name,
            WebIdentityToken=self.access_token,
        )
        print(
            f'[MinioS3::_sts_assume_role] Role assumed. Session name is "{role_session_name}".'
        )
        return sts_role

    def _create_s3_resource(self):
        print(
            f'[MinioS3::_create_s3_resource] Trying to create an S3 client for endpoint "{self.endpoint_url}"...'
        )
        s3 = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.sts_creds["AccessKeyId"],
            aws_secret_access_key=self.sts_creds["SecretAccessKey"],
            aws_session_token=self.sts_creds["SessionToken"],
            config=Config(signature_version="s3v4"),
        )

        print(f"[MinioS3::_create_s3_resource] S3 client created.")
        return s3

    def download_file(self, bucket_name, object_path, output_path):
        try:
            self.s3.download_file(bucket_name, object_path, output_path)
        except botocore.exceptions.ClientError as e:
            print("Error downloading the Object from S3-Bucket!")
            raise e


class User:
    def __init__(
        self,
        username,
        password,
        keycloak_endpoint_url,
        keycloak_client_id,
        keycloak_realm_name,
        keycloak_client_secret_key=None,
    ):
        self.username = username
        self.password = password

        self.keycloak_endpoint_url = keycloak_endpoint_url
        self.keycloak_client_id = keycloak_client_id
        self.keycloak_realm_name = keycloak_realm_name
        self.keycloak_client_secret_key = keycloak_client_secret_key

        self.access_token = None
        self.refresh_token = None
        self.minios3 = None

    def authenticate(self):
        print(
            f'[User::_authenticate] Trying to authenticate user "{self.username}" with password "{self.password}"...'
        )
        oid = OpenID(
            server_url=self.keycloak_endpoint_url,
            client_id=self.keycloak_client_id,
            realm_name=self.keycloak_realm_name,
            username=self.username,
            password=self.password,
            client_secret_key=self.keycloak_client_secret_key,
        )
        access_token, refresh_token = oid.get_tokens()
        print(
            f'[User::_authenticate] User "{self.username}" successfully acquired an access token: "{access_token[:5]}...{access_token[-5:]}"'
        )

        self.access_token = access_token
        self.refresh_token = refresh_token

    def minios3_init(self, endpoint_url, use_ssl=True):
        assert self.access_token and self.refresh_token
        self.minios3 = MinioS3(
            endpoint_url=endpoint_url,
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            use_ssl=use_ssl,
        )


def main():
    user = User(
        username=ARGS.username,
        password=ARGS.password,
        keycloak_endpoint_url=ARGS.keycloak_endpoint_url,
        keycloak_client_id=ARGS.keycloak_client_id,
        keycloak_realm_name=ARGS.keycloak_realm_name,
    )
    user.authenticate()
    user.minios3_init(endpoint_url=ARGS.minio_endpoint_url, use_ssl=(not ARGS.no_ssl))

    path_parts = ARGS.object_path.split("/")
    bucket_name = path_parts[0]
    object_key = "/".join(path_parts[1:])
    user.minios3.download_file(bucket_name, object_key, ARGS.output_path)


if __name__ == "__main__":
    main()

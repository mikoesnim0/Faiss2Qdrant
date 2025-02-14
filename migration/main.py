"""
FAISS to Qdrant Migration Script
This script handles the migration of vector data from FAISS to Qdrant.
"""

import os
import logging
from typing import List
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_environment():
    """환경 변수 로드 및 검증"""
    load_dotenv()
    
    required_vars = [
        "FAISS_VECTORSTORE_PATH",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "COLLECTION_NAME",
        "OPENAI_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    return {var: os.getenv(var) for var in required_vars}

def load_faiss_store_s3(path: str, openai_api_key: str):
    """FAISS 벡터스토어 로드"""
    dir_path = "./vectorstore"
    s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    s3.download_file(Bucket=os.getenv('AWS_BUCKET_NAME'), Key=path + "/faiss.index", Filename=dir_path + "/faiss.index")
    s3.download_file(Bucket=os.getenv('AWS_BUCKET_NAME'), Key=path + "/faiss.pkl", Filename=dir_path + "/faiss.pkl")

    return load_faiss_store_local(dir_path, openai_api_key, allow_dangerous_deserialization=True)

def upload_faiss_store_s3(path: str, openai_api_key: str):
    """FAISS 벡터스토어 업로드"""
    dir_path = "./vectorstore"
    s3 = boto3.client('s3', aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'))
    s3.upload_file(Bucket=os.getenv('AWS_BUCKET_NAME'), Key=path + "/faiss.index", Filename=dir_path + "/faiss.index")
    s3.upload_file(Bucket=os.getenv('AWS_BUCKET_NAME'), Key=path + "/faiss.pkl", Filename=dir_path + "/faiss.pkl")
    # make metadata.json for making new vectorstore version
    with open(dir_path + "/metadata.json", "r") as f:
        metadata = json.load(f)

    # append version to metadata.json with checking if version is already in metadata.json
    # also need to append metadata list by version
    if "version" in metadata:
        metadata["version"] += 1
    else:
        metadata["version"] = 1
    metadata["created_at"] = datetime.now().isoformat()
    metadata["metadata_list"].append(metadata)
    with open(dir_path + "/metadata.json", "w") as f:
        json.dump(metadata, f)
    s3.upload_file(Bucket=os.getenv('AWS_BUCKET_NAME'), Key=path + "/metadata.json", Filename=dir_path + "/metadata.json")

def load_faiss_store_local(path: str, openai_api_key: str):
    """FAISS 벡터스토어 로드"""
    if not all(os.path.exists(os.path.join(path, f)) for f in ["index.faiss", "index.pkl"]):
        raise FileNotFoundError(f"Required FAISS files not found in {path}")
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    
    try:
        return FAISS.load_local(
            folder_path=path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise Exception(f"Failed to load FAISS vectorstore: {str(e)}")

def setup_qdrant_collection(client: QdrantClient, collection_name: str, vector_size: int = 1536):
    """Qdrant 컬렉션 설정"""
    try:
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created collection: {collection_name}")
    except Exception as e:
        raise Exception(f"Failed to create Qdrant collection: {str(e)}")

def convert_and_upload_vectors(
    faiss_store,
    qdrant_client: QdrantClient,
    collection_name: str,
    batch_size: int = 500
):
    """FAISS 벡터를 Qdrant로 변환 및 업로드"""
    total_vectors = len(faiss_store.index_to_docstore_id)
    points = []
    total_uploaded = 0
    
    for i in tqdm(range(total_vectors), desc="Converting and uploading vectors"):
        doc_id = faiss_store.index_to_docstore_id[i]
        doc = faiss_store.docstore.search(doc_id)
        vector = faiss_store.index.reconstruct(i)
        
        point = PointStruct(
            id=doc_id,
            vector=vector.tolist(),
            payload=doc.metadata
        )
        points.append(point)
        
        # 배치 크기에 도달하면 업로드
        if len(points) >= batch_size:
            try:
                qdrant_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                total_uploaded += len(points)
                logger.info(f"Uploaded {total_uploaded}/{total_vectors} vectors")
                points = []
            except Exception as e:
                raise Exception(f"Failed to upload batch to Qdrant: {str(e)}")
    
    # 남은 포인트 업로드
    if points:
        try:
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points
            )
            total_uploaded += len(points)
            logger.info(f"Uploaded {total_uploaded}/{total_vectors} vectors")
        except Exception as e:
            raise Exception(f"Failed to upload final batch to Qdrant: {str(e)}")
    
    return total_uploaded

def main():
    try:
        # 1. 환경 변수 로드
        logger.info("Loading environment variables...")
        env = load_environment()
        
        # 2. FAISS 벡터스토어 로드
        input("Which vectorstore do you want to load? : AWS S3 or Local")
        if input == "AWS S3":
            env['FAISS_VECTORSTORE_PATH'] = 

        logger.info(f"Loading FAISS vectorstore from {env['FAISS_VECTORSTORE_PATH']}")
        faiss_store = load_faiss_store(
            env['FAISS_VECTORSTORE_PATH'],
            env['OPENAI_API_KEY']
        )
        
        # 3. Qdrant 클라이언트 설정
        logger.info("Initializing Qdrant client...")
        qdrant_client = QdrantClient(
            url=env['QDRANT_URL'],
            api_key=env['QDRANT_API_KEY']
        )
        
        # 4. Qdrant 컬렉션 생성
        logger.info(f"Setting up Qdrant collection: {env['COLLECTION_NAME']}")
        setup_qdrant_collection(qdrant_client, env['COLLECTION_NAME'])
        
        # 5. 데이터 변환 및 업로드
        logger.info("Starting vector migration...")
        total_uploaded = convert_and_upload_vectors(
            faiss_store,
            qdrant_client,
            env['COLLECTION_NAME']
        )
        
        logger.info(f"Migration completed successfully! Total vectors uploaded: {total_uploaded}")
        
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
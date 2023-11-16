from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import json

from langchain.schema.document import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore


class IDOL(VectorStore):
    """`IDOL` vector store.
    """

    def __init__(
        self,
        embedding: Embeddings,
        url: Optional[str] = 'http://localhost:9100',
        vector_field: Optional[str] = 'VECTOR',
        database: Optional[str] = 'DOCQA',
        index_batch_size: Optional[int] = 5*1204*1024,
        vector_search: Optional[bool] = True,
    ):
        """Initialize with necessary components."""
        self.embedding = embedding
        self.url = url
        self.vector_field = vector_field
        self.database = database
        self.index_batch_size = index_batch_size
        self.vector_search = vector_search


    def _index(self, content: str):
        """
        index to IDOL Content

        Parameters
        ----------
        content : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        headers = {'Content-Type': 'text/plain; charset=UTF-8'}
        data = f'{content}\n#DREENDDATAREFERENCE'.encode('utf-8')
        res = requests.post(f'{self.url}/DREADDDATA?CreateDatabase=true',
                            data = data,
                            headers = headers)
        if res.status_code != 200:
            print(f'ERROR: {res.text}')


    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        ids = []

        embeddings = self.embedding.embed_documents(list(texts))
        last_source = None
        section = -1
        index = ''
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            source = metadata['source'] if 'source' in metadata else f'unkown_{i}'
            section = section + 1 if source == last_source else 0
            if index and self.index_batch_size < len(index):
                self._index(index)
                index = ''
            last_source = source
            ids.append(f'{source}#{section}')
            vector = ','.join([str(x) for x in embeddings[i]])
            content = texts[i]
            index = f"""{index}
#DREREFERENCE {source}
#DREFIELD {self.vector_field}="{vector}"
#DRESECTION {section}
#DREDBNAME {self.database}
#DRECONTENT
{content}
#DREENDDOC
"""
        
        if index:
            self._index(index)
            index = ''

        return ids

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of documents most similar to the query with distance.
        """
        docs_with_scores: List[Tuple[Document, float]] = []

        # TODO: search doc and score from IDOL Content
        #if self.vector_search:
        #   query_embedding = self.embedding.embed_query(query)

        return docs_with_scores


    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query.
        """
        # search doc from IDOL Content
        text = ''
        if self.vector_search:
            query_embedding = self.embedding.embed_query(query)
            vector = ','.join([str(x) for x in query_embedding])
            text = 'text=VECTOR{' + vector + '}:VECTOR'
        else:
            text = f'DetectLanguageType=true&anylanguage=true&text={query}'
        url = f'{self.url}/a=query&ResponseFormat=json&maxresults={k}&{text}'

        try:
            res = requests.get(url)
        except Exception as e:
            print(f'ERROR: {e}')
            return []

        jres = json.loads(res.text)
        res_data = jres['autnresponse']['responsedata']
        if not 'autn:numhits' in res_data:
            print(f'ERROR: {res.text}\nrequest URL:\n{url}')
            return []
        elif 0 == int(res_data['autn:numhits']['$']):
            return []

        return [Document(page_content=c['DRECONTENT'][0]['$'],
                         metadata={'source': hit['autn:reference']['$']})
                for hit in res_data['autn:hit']
                for c in hit['autn:content']['DOCUMENT'] ]


    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict]] = None,
        url: Optional[str] = 'http://localhost:9101',
        vector_field: Optional[str] = 'VECTOR',
        database: Optional[str] = 'DOCQA',
        index_batch_size: Optional[int] = 5*1204*1024,
        vector_search: Optional[bool] = True,
        **kwargs: Any,
    ) -> IDOL:
        """Construct IDOL wrapper from raw documents.
        This is a user friendly interface that:
            1. Initializes a index
            2. Index to IDOL Content
        This is intended to be a quick way to get started.
        """
        idol = cls(embedding, url, vector_field, database, index_batch_size,
                   vector_search);
        idol.add_texts(texts, metadatas)
        return idol

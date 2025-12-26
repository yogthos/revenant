"""ChromaDB storage for sentence templates."""

import hashlib
import json
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from ..config import ChromaDBConfig
from ..utils.logging import get_logger
from ..utils.nlp import split_into_sentences
from .models import (
    SentenceTemplate,
    CorpusStatistics,
    VocabularyProfile,
    SentenceType,
    RhetoricalRole,
    LogicalRelation,
)
from .extractor import SkeletonExtractor
from .statistics import CorpusStatisticsExtractor

logger = get_logger(__name__)


@dataclass
class TemplateStoreConfig:
    """Configuration for template storage."""
    collection_name: str = "sentence_templates"
    stats_collection_name: str = "corpus_statistics"
    vocab_collection_name: str = "vocabulary_profiles"


class TemplateStore:
    """ChromaDB storage for sentence templates.

    Manages:
    - sentence_templates: Individual sentence skeletons with metadata
    - corpus_statistics: Aggregated statistics per author
    - vocabulary_profiles: Author vocabulary profiles
    """

    def __init__(
        self,
        config: Optional[ChromaDBConfig] = None,
        store_config: Optional[TemplateStoreConfig] = None
    ):
        """Initialize template store.

        Args:
            config: ChromaDB configuration.
            store_config: Template store configuration.
        """
        self.config = config or ChromaDBConfig()
        self.store_config = store_config or TemplateStoreConfig()
        self._client = None
        self._templates_collection = None
        self._stats_collection = None
        self._vocab_collection = None

        # Extractors
        self.skeleton_extractor = SkeletonExtractor()
        self.stats_extractor = CorpusStatisticsExtractor()

    @property
    def client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            import chromadb
            from chromadb.config import Settings

            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )

            if self.config.persist_path:
                self._client = chromadb.PersistentClient(
                    path=self.config.persist_path,
                    settings=settings
                )
                logger.info(f"Template store initialized at: {self.config.persist_path}")
            else:
                self._client = chromadb.Client(settings=settings)
                logger.info("Template store initialized in-memory")

        return self._client

    @property
    def templates_collection(self):
        """Get or create the sentence_templates collection."""
        if self._templates_collection is None:
            self._templates_collection = self.client.get_or_create_collection(
                name=self.store_config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._templates_collection

    @property
    def stats_collection(self):
        """Get or create the corpus_statistics collection."""
        if self._stats_collection is None:
            self._stats_collection = self.client.get_or_create_collection(
                name=self.store_config.stats_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._stats_collection

    @property
    def vocab_collection(self):
        """Get or create the vocabulary_profiles collection."""
        if self._vocab_collection is None:
            self._vocab_collection = self.client.get_or_create_collection(
                name=self.store_config.vocab_collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._vocab_collection

    def index_corpus(
        self,
        paragraphs: List[str],
        author: str,
        document_id: str = ""
    ) -> Dict[str, int]:
        """Index an author's corpus.

        Extracts templates from all sentences and computes statistics.

        Args:
            paragraphs: List of paragraph texts.
            author: Author name.
            document_id: Optional document identifier.

        Returns:
            Dictionary with indexing statistics.
        """
        templates_indexed = 0
        total_sentences = 0

        logger.info(f"Indexing corpus for author '{author}': {len(paragraphs)} paragraphs")

        # Extract and index templates
        for para_idx, paragraph in enumerate(paragraphs):
            para_id = f"{document_id}_{para_idx}" if document_id else str(para_idx)

            templates = self.skeleton_extractor.extract_templates_from_paragraph(
                paragraph, author=author, document_id=para_id
            )

            for template in templates:
                if self._index_template(template):
                    templates_indexed += 1

            total_sentences += len(templates)

        # Extract and store corpus statistics
        stats = self.stats_extractor.extract(paragraphs)
        self._store_statistics(author, stats, document_id)

        # Extract and store vocabulary profile
        vocab = self.stats_extractor.extract_vocabulary_profile(paragraphs)
        self._store_vocabulary(author, vocab, document_id)

        logger.info(
            f"Indexed for '{author}': {templates_indexed} templates from "
            f"{total_sentences} sentences"
        )

        return {
            "templates": templates_indexed,
            "sentences": total_sentences,
            "paragraphs": len(paragraphs),
        }

    def _index_template(self, template: SentenceTemplate) -> bool:
        """Index a single template.

        Args:
            template: Template to index.

        Returns:
            True if indexed successfully.
        """
        try:
            # Check if already exists
            existing = self.templates_collection.get(ids=[template.id])
            if existing and existing['ids']:
                logger.debug(f"Template already exists: {template.id}")
                return False

            # Use skeleton as the document for embedding similarity
            # This allows finding similar syntactic structures
            self.templates_collection.add(
                ids=[template.id],
                documents=[template.skeleton],
                metadatas=[template.to_chromadb_metadata()]
            )
            return True

        except Exception as e:
            logger.error(f"Failed to index template {template.id}: {e}")
            return False

    def _store_statistics(
        self,
        author: str,
        stats: CorpusStatistics,
        document_id: str = ""
    ) -> bool:
        """Store corpus statistics for an author.

        Args:
            author: Author name.
            stats: Computed statistics.
            document_id: Optional document identifier.

        Returns:
            True if stored successfully.
        """
        try:
            stats_id = f"stats_{author}_{document_id}" if document_id else f"stats_{author}"

            # Convert to JSON for storage
            stats_dict = stats.to_dict()

            # Store with author as document for retrieval
            self.stats_collection.upsert(
                ids=[stats_id],
                documents=[author],
                metadatas=[{
                    "author": author,
                    "document_id": document_id,
                    "stats_json": json.dumps(stats_dict),
                }]
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store statistics for {author}: {e}")
            return False

    def _store_vocabulary(
        self,
        author: str,
        vocab: VocabularyProfile,
        document_id: str = ""
    ) -> bool:
        """Store vocabulary profile for an author.

        Args:
            author: Author name.
            vocab: Vocabulary profile.
            document_id: Optional document identifier.

        Returns:
            True if stored successfully.
        """
        try:
            vocab_id = f"vocab_{author}_{document_id}" if document_id else f"vocab_{author}"

            # Convert to JSON for storage
            vocab_dict = vocab.to_dict()

            self.vocab_collection.upsert(
                ids=[vocab_id],
                documents=[author],
                metadatas=[{
                    "author": author,
                    "document_id": document_id,
                    "vocab_json": json.dumps(vocab_dict),
                }]
            )
            return True

        except Exception as e:
            logger.error(f"Failed to store vocabulary for {author}: {e}")
            return False

    def query_templates(
        self,
        query_text: str,
        author: Optional[str] = None,
        sentence_type: Optional[SentenceType] = None,
        rhetorical_role: Optional[RhetoricalRole] = None,
        logical_relation: Optional[LogicalRelation] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
        min_word_count: Optional[int] = None,
        max_word_count: Optional[int] = None,
        n_results: int = 10
    ) -> List[SentenceTemplate]:
        """Query templates with optional filters.

        Args:
            query_text: Text to find similar templates for.
            author: Filter by author.
            sentence_type: Filter by position type (OPENER/BODY/CLOSER).
            rhetorical_role: Filter by rhetorical function.
            logical_relation: Filter by logical relation to previous.
            min_complexity: Minimum complexity score.
            max_complexity: Maximum complexity score.
            min_word_count: Minimum word count.
            max_word_count: Maximum word count.
            n_results: Number of results to return.

        Returns:
            List of matching SentenceTemplates.
        """
        filter_desc = []
        if author:
            filter_desc.append(f"author={author}")
        if sentence_type:
            filter_desc.append(f"type={sentence_type.value}")
        if rhetorical_role:
            filter_desc.append(f"role={rhetorical_role.value}")

        logger.debug(f"Querying templates: {', '.join(filter_desc) if filter_desc else 'no filters'}")

        where_filter = self._build_template_filter(
            author=author,
            sentence_type=sentence_type,
            rhetorical_role=rhetorical_role,
            logical_relation=logical_relation,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
            min_word_count=min_word_count,
            max_word_count=max_word_count,
        )

        results = self.templates_collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_filter
        )

        templates = self._results_to_templates(results)
        logger.debug(f"  Query returned {len(templates)} templates")

        return templates

    def get_templates_for_sequence(
        self,
        author: str,
        sentence_types: List[SentenceType],
        rhetorical_roles: List[RhetoricalRole],
        n_candidates: int = 5
    ) -> List[List[SentenceTemplate]]:
        """Get template candidates for a sequence of sentence requirements.

        Args:
            author: Author to retrieve from.
            sentence_types: List of required sentence types.
            rhetorical_roles: List of required rhetorical roles.
            n_candidates: Number of candidates per position.

        Returns:
            List of [candidate templates] for each position.
        """
        candidates = []

        for sent_type, role in zip(sentence_types, rhetorical_roles):
            position_candidates = self.query_templates(
                query_text="",  # No semantic query, just filter
                author=author,
                sentence_type=sent_type,
                rhetorical_role=role,
                n_results=n_candidates
            )
            candidates.append(position_candidates)

        return candidates

    def get_statistics(self, author: str) -> Optional[CorpusStatistics]:
        """Get corpus statistics for an author.

        Args:
            author: Author name.

        Returns:
            CorpusStatistics or None if not found.
        """
        results = self.stats_collection.get(
            where={"author": author},
            limit=1
        )

        if not results or not results['ids']:
            return None

        metadata = results['metadatas'][0]
        stats_dict = json.loads(metadata.get('stats_json', '{}'))

        return CorpusStatistics.from_dict(stats_dict)

    def get_vocabulary(self, author: str) -> Optional[VocabularyProfile]:
        """Get vocabulary profile for an author.

        Args:
            author: Author name.

        Returns:
            VocabularyProfile or None if not found.
        """
        results = self.vocab_collection.get(
            where={"author": author},
            limit=1
        )

        if not results or not results['ids']:
            return None

        metadata = results['metadatas'][0]
        vocab_dict = json.loads(metadata.get('vocab_json', '{}'))

        return VocabularyProfile.from_dict(vocab_dict)

    def get_transition_probability(
        self,
        author: str,
        from_role: RhetoricalRole,
        to_role: RhetoricalRole
    ) -> float:
        """Get transition probability between rhetorical roles.

        Args:
            author: Author name.
            from_role: Source role.
            to_role: Target role.

        Returns:
            Transition probability (0-1).
        """
        stats = self.get_statistics(author)
        if not stats:
            return 0.0

        return stats.get_transition_probability(from_role.value, to_role.value)

    def get_likely_next_roles(
        self,
        author: str,
        current_role: RhetoricalRole,
        top_k: int = 3
    ) -> List[Tuple[RhetoricalRole, float]]:
        """Get most likely next rhetorical roles.

        Args:
            author: Author name.
            current_role: Current sentence's role.
            top_k: Number of top roles to return.

        Returns:
            List of (role, probability) tuples.
        """
        stats = self.get_statistics(author)
        if not stats:
            return []

        transitions = stats.get_likely_next_roles(current_role.value, top_k)

        result = []
        for role_str, prob in transitions:
            try:
                role = RhetoricalRole(role_str)
                result.append((role, prob))
            except ValueError:
                continue

        return result

    def _build_template_filter(
        self,
        author: Optional[str] = None,
        sentence_type: Optional[SentenceType] = None,
        rhetorical_role: Optional[RhetoricalRole] = None,
        logical_relation: Optional[LogicalRelation] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
        min_word_count: Optional[int] = None,
        max_word_count: Optional[int] = None,
    ) -> Optional[Dict]:
        """Build ChromaDB where filter for templates.

        Args:
            Various filter parameters.

        Returns:
            ChromaDB where filter or None.
        """
        conditions = []

        if author:
            conditions.append({"author": author})

        if sentence_type:
            conditions.append({"sentence_type": sentence_type.value})

        if rhetorical_role:
            conditions.append({"rhetorical_role": rhetorical_role.value})

        if logical_relation:
            conditions.append({"logical_relation": logical_relation.value})

        if min_complexity is not None:
            conditions.append({"complexity_score": {"$gte": min_complexity}})

        if max_complexity is not None:
            conditions.append({"complexity_score": {"$lte": max_complexity}})

        if min_word_count is not None:
            conditions.append({"word_count": {"$gte": min_word_count}})

        if max_word_count is not None:
            conditions.append({"word_count": {"$lte": max_word_count}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _results_to_templates(self, results: Dict) -> List[SentenceTemplate]:
        """Convert ChromaDB results to SentenceTemplate objects.

        Args:
            results: Raw ChromaDB results.

        Returns:
            List of SentenceTemplate objects.
        """
        templates = []

        if not results or not results.get('ids'):
            return templates

        for i, id_ in enumerate(results['ids'][0]):
            metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
            document = results['documents'][0][i] if results.get('documents') else ""

            try:
                template = SentenceTemplate(
                    id=id_,
                    skeleton=metadata.get('skeleton', document),
                    pos_pattern=metadata.get('pos_pattern', ''),
                    word_count=metadata.get('word_count', 0),
                    complexity_score=metadata.get('complexity_score', 0.0),
                    clause_count=metadata.get('clause_count', 1),
                    sentence_type=SentenceType(metadata.get('sentence_type', 'body')),
                    rhetorical_role=RhetoricalRole(metadata.get('rhetorical_role', 'elaboration')),
                    logical_relation=LogicalRelation(metadata.get('logical_relation', 'additive')),
                    original_text="",  # Not stored in metadata
                    author=metadata.get('author', ''),
                    document_id=metadata.get('document_id', ''),
                )
                templates.append(template)
            except Exception as e:
                logger.warning(f"Failed to parse template {id_}: {e}")

        return templates

    def get_stats(self) -> Dict[str, int]:
        """Get collection statistics.

        Returns:
            Dictionary with collection counts.
        """
        return {
            "templates": self.templates_collection.count(),
            "statistics": self.stats_collection.count(),
            "vocabularies": self.vocab_collection.count(),
        }

    def clear_author(self, author: str) -> int:
        """Clear all data for an author.

        Args:
            author: Author to clear.

        Returns:
            Number of items deleted.
        """
        deleted = 0

        # Delete templates
        try:
            results = self.templates_collection.get(where={"author": author})
            if results and results['ids']:
                self.templates_collection.delete(ids=results['ids'])
                deleted += len(results['ids'])
        except Exception as e:
            logger.error(f"Error deleting templates for {author}: {e}")

        # Delete statistics
        try:
            results = self.stats_collection.get(where={"author": author})
            if results and results['ids']:
                self.stats_collection.delete(ids=results['ids'])
                deleted += len(results['ids'])
        except Exception as e:
            logger.error(f"Error deleting stats for {author}: {e}")

        # Delete vocabulary
        try:
            results = self.vocab_collection.get(where={"author": author})
            if results and results['ids']:
                self.vocab_collection.delete(ids=results['ids'])
                deleted += len(results['ids'])
        except Exception as e:
            logger.error(f"Error deleting vocab for {author}: {e}")

        logger.info(f"Cleared {deleted} items for author '{author}'")
        return deleted

    def clear_all(self) -> None:
        """Clear all collections."""
        try:
            self.client.delete_collection(self.store_config.collection_name)
        except Exception:
            pass

        try:
            self.client.delete_collection(self.store_config.stats_collection_name)
        except Exception:
            pass

        try:
            self.client.delete_collection(self.store_config.vocab_collection_name)
        except Exception:
            pass

        self._templates_collection = None
        self._stats_collection = None
        self._vocab_collection = None
        logger.info("Cleared all template collections")

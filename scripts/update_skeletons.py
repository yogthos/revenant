#!/usr/bin/env python3
"""Update existing ChromaDB chunks with rhetorical skeletons.

This script:
1. Loads all chunks from ChromaDB for a given author
2. Extracts rhetorical skeletons using DeepSeek
3. Updates the metadata with the skeleton

Usage:
    python scripts/update_skeletons.py --author "H.P. Lovecraft"
    python scripts/update_skeletons.py --all  # Update all authors
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.corpus_indexer import get_indexer
from src.rag.skeleton_extractor import extract_skeleton
from src.config import load_config
from src.llm.provider import create_critic_provider
from src.utils.logging import get_logger

logger = get_logger(__name__)


def update_author_skeletons(author: str, llm_provider, batch_size: int = 10) -> int:
    """Update skeletons for all chunks of an author.

    Args:
        author: Author name.
        llm_provider: LLM provider for skeleton extraction.
        batch_size: Number of chunks to process before saving.

    Returns:
        Number of chunks updated.
    """
    indexer = get_indexer()
    collection = indexer.collection

    # Get all chunks for this author
    results = collection.get(
        where={"author": author},
        include=["documents", "metadatas"]
    )

    ids = results.get("ids", [])
    documents = results.get("documents", [])
    metadatas = results.get("metadatas", [])

    if not ids:
        logger.warning(f"No chunks found for author: {author}")
        return 0

    logger.info(f"Found {len(ids)} chunks for {author}")

    # Check how many already have skeletons
    need_update = []
    for i, (chunk_id, doc, meta) in enumerate(zip(ids, documents, metadatas)):
        if not meta.get("skeleton"):
            need_update.append((chunk_id, doc, meta))

    if not need_update:
        logger.info(f"All {len(ids)} chunks already have skeletons")
        return 0

    logger.info(f"{len(need_update)} chunks need skeleton extraction")

    # Process in batches
    updated = 0
    try:
        from tqdm import tqdm
        iterator = tqdm(need_update, desc=f"Extracting skeletons for {author}")
    except ImportError:
        iterator = need_update

    batch_ids = []
    batch_metas = []

    for chunk_id, doc, meta in iterator:
        # Extract skeleton
        skeleton = extract_skeleton(doc, llm_provider)

        if skeleton.moves:
            # Update metadata
            meta["skeleton"] = skeleton.to_metadata()
            batch_ids.append(chunk_id)
            batch_metas.append(meta)
            updated += 1

            # Save batch
            if len(batch_ids) >= batch_size:
                collection.update(ids=batch_ids, metadatas=batch_metas)
                logger.debug(f"Saved batch of {len(batch_ids)} skeletons")
                batch_ids = []
                batch_metas = []

    # Save remaining
    if batch_ids:
        collection.update(ids=batch_ids, metadatas=batch_metas)
        logger.debug(f"Saved final batch of {len(batch_ids)} skeletons")

    logger.info(f"Updated {updated} chunks with skeletons for {author}")
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Update ChromaDB chunks with rhetorical skeletons"
    )
    parser.add_argument(
        "--author",
        type=str,
        help="Author name to update (e.g., 'H.P. Lovecraft')"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Update all indexed authors"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for saving (default: 10)"
    )

    args = parser.parse_args()

    if not args.author and not args.all:
        parser.error("Must specify --author or --all")

    # Load config and create LLM provider
    config = load_config()
    llm_provider = create_critic_provider(config.llm)

    indexer = get_indexer()

    if args.all:
        authors = indexer.get_authors()
        if not authors:
            logger.error("No authors found in index")
            return 1

        logger.info(f"Updating skeletons for {len(authors)} authors: {authors}")
        total = 0
        for author in authors:
            updated = update_author_skeletons(author, llm_provider, args.batch_size)
            total += updated

        logger.info(f"Total chunks updated: {total}")

    else:
        updated = update_author_skeletons(args.author, llm_provider, args.batch_size)
        if updated == 0:
            logger.info("No updates needed")

    return 0


if __name__ == "__main__":
    sys.exit(main())

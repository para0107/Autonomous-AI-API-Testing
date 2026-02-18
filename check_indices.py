"""
Check what's actually in the vector store indices
"""
import asyncio
from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingManager


async def check_indices():
    print("=" * 60)
    print("CHECKING VECTOR STORE INDICES")
    print("=" * 60)

    vs = VectorStore()

    # Check each index
    for index_name in ['test_patterns', 'edge_cases', 'validation_rules']:
        print(f"\n📁 Index: {index_name}")
        print("-" * 60)

        try:
            stats = vs.get_index_stats(index_name)
            print(f"Total embeddings: {stats['total_embeddings']}")
            print(f"Dimension: {stats['dimension']}")
            print(f"Index type: {stats['index_type']}")
            print(f"Metadata count: {stats['metadata_count']}")

            # Try to get some metadata
            if stats['metadata_count'] > 0:
                metadata_store = vs.metadata_stores.get(index_name, {})
                sample_ids = list(metadata_store.keys())[:3]
                print(f"\nSample IDs: {sample_ids}")
                for id_ in sample_ids:
                    meta = metadata_store[id_]
                    print(f"  ID {id_}: {meta.get('title', 'No title')[:50]}")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Now try a search with a very simple query
    print("\n" + "=" * 60)
    print("TESTING SEARCH WITH SIMPLE QUERY")
    print("=" * 60)

    em = EmbeddingManager()
    simple_queries = [
        "user",
        "reservation",
        "GET"
    ]

    for query in simple_queries:
        print(f"\nQuery: '{query}'")
        embedding = await em.embed_text(query)

        for idx_name in ['test_patterns']:
            try:
                ids, distances, metadata = vs.search(idx_name, embedding, k=3)
                valid = [m for m in metadata if m]
                print(f"  {idx_name}: {len(valid)} results")
                if valid:
                    print(f"    Best match: {valid[0].get('title', 'No title')[:60]}")
            except Exception as e:
                print(f"  Error: {e}")


if __name__ == "__main__":
    asyncio.run(check_indices())
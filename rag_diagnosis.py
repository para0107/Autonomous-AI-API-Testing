"""
RAG Diagnosis - Direct test (no pytest needed)
"""
import asyncio
from rag.embeddings import EmbeddingManager
from rag.vector_store import VectorStore


async def test():
    print("="*60)
    print("RAG DIAGNOSIS TEST")
    print("="*60)

    em = EmbeddingManager()
    vs = VectorStore()

    query = "GET reservation by ID POST create reservation"
    print(f"\n🔍 Searching for: {query}\n")

    embedding = await em.embed_text(query)
    print(f"✅ Generated embedding (dim: {len(embedding)})\n")

    total_results = 0

    for idx_name in ['test_patterns', 'edge_cases', 'validation_rules']:
        print(f"\n📁 Index: {idx_name}")
        print("-" * 60)

        try:
            ids, distances, metadata = vs.search(idx_name, embedding, k=5)
            valid_results = [m for m in metadata if m]
            total_results += len(valid_results)

            print(f"Found {len(valid_results)} results:")

            for i, (dist, meta) in enumerate(zip(distances[:5], metadata[:5]), 1):
                if meta:
                    title = meta.get('title', 'No title')
                    similarity = max(0, 1 - dist)
                    print(f"  {i}. [{dist:.3f}|{similarity:.1%}] {title[:70]}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "="*60)
    print(f"TOTAL RESULTS FOUND: {total_results}")
    print("="*60)

    if total_results == 0:
        print("\n⚠️  WARNING: RAG returned 0 results!")
        print("This explains why the pipeline shows 'No similar tests found'")
    else:
        print(f"\n✅ SUCCESS: RAG is working! Found {total_results} relevant examples")


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test())
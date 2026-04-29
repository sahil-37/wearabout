const BASE = '/api/v1'

export async function getRecommendations(file, gender = 'unisex', topK = 20) {
  const form = new FormData()
  form.append('file', file)

  const params = new URLSearchParams({ gender })
  if (topK) params.set('top_k', topK)

  const res = await fetch(`${BASE}/recommend?${params}`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function findSimilar(file, topK = 20) {
  const form = new FormData()
  form.append('file', file)

  const params = new URLSearchParams()
  if (topK) params.set('top_k', topK)

  const res = await fetch(`${BASE}/find-similar?${params}`, {
    method: 'POST',
    body: form,
  })

  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail || `HTTP ${res.status}`)
  }
  return res.json()
}

export async function healthCheck() {
  const res = await fetch(`${BASE}/health`)
  return res.json()
}

import type { PredictionResponse } from '../types/prediction'

const API_URL =
  import.meta.env.VITE_API_URL || 'http://localhost:8000'

export async function classifyImage(
  file: File
): Promise<PredictionResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    body: formData,
  })

  const body = await response.json().catch(() => null)

  if (!response.ok) {
    if (response.status === 400 || response.status === 415) {
      throw new Error(
        'Invalid image. Please upload a JPG, JPEG, or PNG file.'
      )
    }

    if (response.status === 503) {
      throw new Error('Classification model is currently unavailable.')
    }

    throw new Error(
      body?.detail || 'Classification could not be completed.'
    )
  }

  return body as PredictionResponse
}

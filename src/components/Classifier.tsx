import { useEffect, useState } from 'react'
import {
  ArrowUpRight,
  ImagePlus,
  ShieldAlert,
  X,
} from 'lucide-react'

import { classifyImage } from '../services/api'
import type { PredictionResponse } from '../types/prediction'
import ResultPanel from './ResultPanel'

function ImagePreview({ source }: { source: string }) {
  return (
    <div className="image-preview">
      <img src={source} alt="Uploaded wound evidence" />

      <span className="corner tl" />
      <span className="corner tr" />
      <span className="corner bl" />
      <span className="corner br" />
    </div>
  )
}

function Classifier() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [status, setStatus] = useState<
    'idle' | 'running' | 'error'
  >('idle')
  const [error, setError] = useState('')

  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview)
      }
    }
  }, [preview])

  const pick = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    const selectedFile = event.target.files?.[0]

    if (!selectedFile) {
      return
    }

    const allowedTypes = [
      'image/jpeg',
      'image/png',
    ]

    if (!allowedTypes.includes(selectedFile.type)) {
      setError(
        'Invalid image. Please upload a JPG, JPEG, or PNG file.'
      )
      return
    }

    if (preview) {
      URL.revokeObjectURL(preview)
    }

    setFile(selectedFile)
    setPreview(URL.createObjectURL(selectedFile))
    setResult(null)
    setStatus('idle')
    setError('')
  }

  const remove = () => {
    if (preview) {
      URL.revokeObjectURL(preview)
    }

    setFile(null)
    setPreview(null)
    setResult(null)
    setStatus('idle')
    setError('')
  }

  const runClassification = async () => {
    if (!file) {
      return
    }

    setStatus('running')
    setError('')
    setResult(null)

    try {
      const prediction = await classifyImage(file)

      setResult(prediction)
      setStatus('idle')
    } catch (cause) {
      setStatus('error')

      const message =
        cause instanceof Error
          ? cause.message
          : 'Classification could not be completed.'

      setError(
        message === 'Failed to fetch'
          ? 'Unable to connect to the classification service.'
          : message
      )
    }
  }

  return (
    <section className="workstation" id="classifier">
      <div className="section-head">
        <div>
          <p className="eyebrow"><i /> CLASSIFICATION WORKSPACE</p>
          <h2>
            From image to <em>finding.</em>
          </h2>
        </div>

        <span className="analysis-id">
          ANALYSIS ID&nbsp;&nbsp;
          {result?.analysis_id || 'PENDING'}
        </span>
      </div>

      <div className="workspace">
        <section className="input-panel">
          <div className="panel-title">
            <span>01 — INPUT IMAGE</span>
            <small>JPG · JPEG · PNG</small>
          </div>

          {file && preview ? (
            <div className="uploaded">
              <ImagePreview source={preview} />

              <div className="uploaded-info">
                <b>{file.name}</b>

                <small>
                  {Math.round(file.size / 1024)} KB · uploaded just now
                </small>

                <button onClick={remove}>
                  <X size={14} />
                  Remove image
                </button>
              </div>
            </div>
          ) : (
            <label className="dropzone">
              <ImagePlus />

              <b>Drop evidence image here</b>

              <span>
                or choose an image from your device
              </span>

              <input
                type="file"
                accept="image/png,image/jpeg"
                onChange={pick}
              />
            </label>
          )}

          <p className="privacy">
            <ShieldAlert size={13} />
            Images are processed for classification and are not intended for
            diagnostic use.
          </p>

          {error && (
            <p className="form-error" role="alert">
              {error}
            </p>
          )}

          <button
            onClick={runClassification}
            className="run"
            disabled={!file || status === 'running'}
          >
            {status === 'running'
              ? 'Analyzing evidence…'
              : 'Run classification'}

            <ArrowUpRight size={16} />
          </button>
        </section>

        <ResultPanel
          result={result}
          status={status}
        />
      </div>

      <p className="disclaimer">
        Predictions are intended for research and educational purposes and
        should not be treated as a medical diagnosis.
      </p>
    </section>
  )
}

export default Classifier

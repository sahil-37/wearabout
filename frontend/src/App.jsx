import { useState, useCallback } from 'react'
import UploadZone from './components/UploadZone'
import PreviewImage from './components/PreviewImage'
import ProductCard from './components/ProductCard'
import GenderFilter from './components/GenderFilter'
import ModeToggle from './components/ModeToggle'
import { getRecommendations, findSimilar } from './api'
import './index.css'

const STATES = { IDLE: 'idle', LOADING: 'loading', DONE: 'done', ERROR: 'error' }

export default function App() {
  const [state, setState]     = useState(STATES.IDLE)
  const [file, setFile]       = useState(null)
  const [preview, setPreview] = useState(null)
  const [gender, setGender]   = useState('unisex')
  const [mode, setMode]       = useState('recommend')
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)

  const items      = result?.recommendations ?? result?.similar_items ?? []
  const detections = result?.detections ?? []

  const handleFile = useCallback((f) => {
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
    setError(null)
    setState(STATES.IDLE)
  }, [])

  async function handleSubmit() {
    if (!file) return
    setState(STATES.LOADING)
    setError(null)
    try {
      const data = mode === 'recommend'
        ? await getRecommendations(file, gender, 20)
        : await findSimilar(file, 20)
      if (!data.success) throw new Error(data.error || 'Unknown error')
      setResult(data)
      setState(STATES.DONE)
    } catch (e) {
      setError(e.message)
      setState(STATES.ERROR)
    }
  }

  function reset() {
    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
    setState(STATES.IDLE)
  }

  return (
    <div className="min-h-screen bg-[#0a0a0a]">

      {/* ── header ── */}
      <header className="border-b border-white/5 sticky top-0 z-20 bg-[#0a0a0a]/90 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-6 h-6 rounded-md bg-white flex items-center justify-center text-black text-xs font-bold shrink-0">
              W
            </div>
            <span className="text-sm font-semibold text-white/90 tracking-tight">
              Wearabout
            </span>
          </div>

          {/* only show controls in header when image is loaded */}
          {preview && (
            <div className="flex items-center gap-3">
              <ModeToggle mode={mode} onChange={(m) => { setMode(m); setResult(null) }} />
              <GenderFilter value={gender} onChange={setGender} />
            </div>
          )}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6">

        {/* ── landing / upload ── */}
        {!preview && (
          <div className="py-24 flex flex-col items-center gap-12">
            <div className="text-center flex flex-col gap-4 max-w-lg">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-white/10 bg-white/5 text-xs text-white/40 font-medium mx-auto">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse inline-block" />
                AI-powered fashion search
              </div>
              <h1 className="text-5xl font-bold text-white leading-[1.1] tracking-tight">
                Find the look,<br />
                <span className="text-white/25">buy the pieces.</span>
              </h1>
              <p className="text-sm text-white/35 leading-relaxed">
                Upload any outfit photo and instantly discover<br />where to buy each item.
              </p>
            </div>

            <div className="w-full max-w-xl flex flex-col gap-3">
              <div className="flex items-center justify-between">
                <ModeToggle mode={mode} onChange={(m) => { setMode(m) }} />
                <GenderFilter value={gender} onChange={setGender} />
              </div>
              <UploadZone onFile={handleFile} disabled={false} />
            </div>
          </div>
        )}

        {/* ── after upload ── */}
        {preview && (
          <div className="py-8 flex flex-col gap-8">

            {/* preview + controls row */}
            <div className="flex gap-8 items-start">

              {/* image preview */}
              <div className="shrink-0">
                <PreviewImage src={preview} detections={detections} />
              </div>

              {/* right side */}
              <div className="flex flex-col gap-6 flex-1 min-w-0 pt-1">

                {/* file info */}
                <div>
                  <p className="text-xs text-white/25 uppercase tracking-wider mb-1.5">Selected file</p>
                  <p className="text-sm font-medium text-white/70 truncate">{file?.name}</p>
                  <p className="text-xs text-white/25 mt-0.5">
                    {file ? (file.size / 1024 / 1024).toFixed(1) + ' MB' : ''}
                  </p>
                </div>

                {/* status */}
                {state === STATES.DONE && (
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
                      <span className="text-sm text-emerald-400 font-medium">
                        {items.length} results found
                      </span>
                    </div>
                    {detections.length > 0 && (
                      <p className="text-xs text-white/25 pl-3.5">
                        {detections.length} clothing item{detections.length !== 1 ? 's' : ''} detected
                      </p>
                    )}
                  </div>
                )}

                {state === STATES.ERROR && (
                  <div className="flex flex-col gap-1 bg-red-500/8 border border-red-500/15 rounded-xl px-4 py-3">
                    <span className="text-xs font-semibold text-red-400 uppercase tracking-wider">
                      Not a fashion image
                    </span>
                    <span className="text-sm text-red-400/60 leading-relaxed">{error}</span>
                  </div>
                )}

                {state === STATES.LOADING && (
                  <div className="flex items-center gap-3">
                    <Spinner />
                    <span className="text-sm text-white/30">Analysing your outfit…</span>
                  </div>
                )}

                {/* actions */}
                <div className="flex items-center gap-3">
                  <button
                    onClick={handleSubmit}
                    disabled={state === STATES.LOADING}
                    className="px-5 py-2 bg-white hover:bg-white/90 disabled:opacity-25 disabled:cursor-not-allowed text-black text-sm font-semibold rounded-xl transition-all"
                  >
                    {state === STATES.LOADING ? 'Analysing…' : 'Get recommendations'}
                  </button>
                  <button
                    onClick={reset}
                    className="px-4 py-2 text-sm text-white/30 hover:text-white/60 bg-white/5 hover:bg-white/8 rounded-xl transition-all border border-white/5"
                  >
                    ← New photo
                  </button>
                </div>
              </div>
            </div>

            {/* divider */}
            {state === STATES.DONE && items.length > 0 && (
              <div className="border-t border-white/5" />
            )}

            {/* results */}
            {state === STATES.DONE && items.length > 0 && (
              <section className="flex flex-col gap-5 pb-12">
                <div className="flex items-baseline gap-3">
                  <h2 className="text-sm font-semibold text-white/70">
                    {mode === 'recommend' ? 'Recommended items' : 'Similar items'}
                  </h2>
                  <span className="text-xs text-white/20">{items.length} results</span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
                  {items.map((item) => (
                    <ProductCard key={item.id} item={item} />
                  ))}
                </div>
              </section>
            )}

            {state === STATES.DONE && items.length === 0 && (
              <div className="text-center py-20">
                <p className="text-3xl mb-3 opacity-20">◎</p>
                <p className="text-sm font-medium text-white/30">No results found</p>
                <p className="text-xs text-white/20 mt-1">
                  {mode === 'recommend'
                    ? 'Try Find Similar mode or use a clearer photo'
                    : 'Try a different image'}
                </p>
              </div>
            )}

          </div>
        )}

      </main>
    </div>
  )
}

function Spinner() {
  return (
    <svg className="animate-spin h-3.5 w-3.5 text-white/30 shrink-0" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}

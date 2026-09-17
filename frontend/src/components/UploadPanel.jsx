import { useEffect, useRef, useState } from 'react'

export default function UploadPanel({ onFileSelect, loading, hasResult, onReset }) {
  const [preview, setPreview] = useState(null)
  const [fileName, setFileName] = useState('')
  const [selectedFile, setSelectedFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [wakingUp, setWakingUp] = useState(false)
  const inputRef = useRef(null)

  // After 5s of loading, show the cold-start message (Render free tier).
  useEffect(() => {
    if (!loading) {
      setWakingUp(false)
      return
    }
    const t = setTimeout(() => setWakingUp(true), 5000)
    return () => clearTimeout(t)
  }, [loading])

  // Revoke old preview URL to avoid memory leaks
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview)
    }
  }, [preview])

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    if (preview) URL.revokeObjectURL(preview)
    setPreview(URL.createObjectURL(file))
    setFileName(file.name)
    setSelectedFile(file)
  }

  const handleAnalyze = () => {
    if (selectedFile && !loading) onFileSelect(selectedFile)
  }

  const handleReset = () => {
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setFileName('')
    setSelectedFile(null)
    if (inputRef.current) inputRef.current.value = ''
    onReset?.()
  }

  return (
    <div className="glass-panel p-4 sm:p-5">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[13.5px] font-semibold text-slate-100 flex items-center gap-2">
          <span className="h-7 w-7 rounded-lg bg-white/[0.06] border border-white/10 grid place-items-center text-[13px]">🖼️</span>
          Upload Image
        </h3>
        <span className="text-[11px] font-normal px-2 py-1 rounded-full bg-white/[0.05] text-slate-400 border border-white/10">224 × 224 • MobileNetV2</span>
      </div>
      <div
        onClick={() => !loading && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          if (!loading) handleFile(e.dataTransfer.files?.[0])
        }}
        className={`border-2 border-dashed rounded-xl p-6 sm:p-8 text-center cursor-pointer transition-all duration-200 ${
          dragOver
            ? 'border-[#e8a04b]/70 bg-[#e8a04b]/[0.07] scale-[1.003]'
            : 'border-white/10 bg-[#0a1220]/50 hover:border-white/15 hover:bg-white/[0.04]'
        } ${loading ? 'opacity-60 pointer-events-none' : ''}`}
      >
        <div className="mx-auto h-11 w-11 rounded-xl bg-white/[0.06] border border-white/10 grid place-items-center text-xl mb-3">📷</div>
        <p className="text-slate-200 text-sm font-medium">
          Drag &amp; drop an image here, or{' '}
          <span className="text-[#e8a04b] font-semibold">
            click to browse
          </span>
        </p>
        <p className="text-slate-500 text-xs mt-1.5">JPG, JPEG or PNG • Max 10 MB</p>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          disabled={loading}
          onChange={(e) => handleFile(e.target.files?.[0])}
        />
      </div>

      {preview && (
        <div className="mt-4">
          <img
            src={preview}
            alt="preview"
            className="w-full max-h-80 object-contain rounded-xl border border-slate-700 shadow-lg shadow-black/40 bg-slate-950 transition-all duration-300"
          />
          {fileName && (
            <p className="text-slate-400 text-xs mt-2 truncate text-center">{fileName}</p>
          )}
        </div>
      )}

      <button
        type="button"
        onClick={handleAnalyze}
        disabled={!selectedFile || loading}
        className={`mt-4 w-full py-3 px-4 rounded-xl text-[13.5px] transition-all duration-200 flex items-center justify-center gap-2 ${
          !selectedFile || loading
            ? 'bg-white/[0.06] text-slate-500 cursor-not-allowed border border-white/10 font-medium'
            : 'btn-primary btn-lift'
        }`}
      >
        {loading && (
          <svg
            className="animate-spin h-5 w-5 text-white"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
        )}
        {loading ? (wakingUp ? 'Warming up the server… (free hosting sleeps when idle, one moment)' : 'Analyzing...') : 'Analyze Image'}
      </button>

      {hasResult && !loading && (
        <button
          type="button"
          onClick={handleReset}
          className="mt-2 w-full py-2 px-4 rounded-xl text-xs sm:text-sm font-semibold text-slate-300 hover:text-white hover:bg-slate-700/60 transition-all duration-300"
        >
          Analyze Another Image
        </button>
      )}
    </div>
  )
}

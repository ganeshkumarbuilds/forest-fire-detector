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
    <div className="bg-slate-800 border border-slate-700 rounded-2xl shadow-xl shadow-black/30 p-4 sm:p-6 transition-all duration-300">
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
        className={`border-2 border-dashed rounded-xl p-6 sm:p-10 text-center cursor-pointer transition-all duration-300 ${
          dragOver
            ? 'border-orange-500 bg-orange-500/10 scale-[1.01]'
            : 'border-slate-600 bg-slate-900/40 hover:border-orange-500/70 hover:bg-slate-700/40'
        } ${loading ? 'opacity-60 pointer-events-none' : ''}`}
      >
        <div className="text-4xl mb-2">📷</div>
        <p className="text-slate-300 text-sm sm:text-base">
          Drag &amp; drop an image here, or{' '}
          <span className="text-orange-400 font-semibold hover:text-orange-300">
            click to browse
          </span>
        </p>
        <p className="text-slate-500 text-xs mt-1">JPG, JPEG or PNG</p>
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
        className={`mt-4 w-full py-3 px-4 rounded-xl font-bold text-sm sm:text-base text-white transition-all duration-300 flex items-center justify-center gap-2 ${
          !selectedFile || loading
            ? 'bg-slate-700 text-slate-400 cursor-not-allowed'
            : 'bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-500 hover:to-red-500 shadow-lg shadow-red-900/40 hover:shadow-red-800/50 active:scale-[0.99]'
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
        {loading ? (wakingUp ? 'Waking up the server, this may take up to a minute on first request...' : 'Analyzing...') : 'Analyze Image'}
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

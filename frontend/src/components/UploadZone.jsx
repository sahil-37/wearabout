import { useRef, useState } from 'react'

export default function UploadZone({ onFile, disabled }) {
  const inputRef = useRef()
  const [dragging, setDragging] = useState(false)

  function handleFiles(files) {
    const file = files?.[0]
    if (file && file.type.startsWith('image/')) onFile(file)
  }

  function onDrop(e) {
    e.preventDefault()
    setDragging(false)
    handleFiles(e.dataTransfer.files)
  }

  return (
    <div
      onClick={() => !disabled && inputRef.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`
        relative flex flex-col items-center justify-center gap-4
        border border-dashed rounded-3xl cursor-pointer select-none
        transition-all duration-300 py-20 px-8
        ${dragging
          ? 'border-white/60 bg-white/5'
          : 'border-white/10 bg-white/[0.02] hover:border-white/25 hover:bg-white/[0.04]'}
        ${disabled ? 'opacity-40 cursor-not-allowed' : ''}
      `}
    >
      {/* icon */}
      <div className={`
        w-16 h-16 rounded-2xl flex items-center justify-center text-2xl
        transition-all duration-300
        ${dragging ? 'bg-white/15 scale-110' : 'bg-white/5'}
      `}>
        📸
      </div>

      <div className="text-center">
        <p className="text-base font-medium text-white/80">
          {dragging ? 'Drop it here' : 'Upload your outfit photo'}
        </p>
        <p className="text-sm text-white/30 mt-1">
          Drag & drop or click to browse · JPG, PNG up to 10 MB
        </p>
      </div>

      <div className="flex items-center gap-6 mt-2 text-xs text-white/20">
        <span>✦ Full body shots work best</span>
        <span>✦ Good lighting helps</span>
      </div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
        disabled={disabled}
      />
    </div>
  )
}

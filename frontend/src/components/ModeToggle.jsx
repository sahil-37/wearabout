const MODES = [
  { value: 'recommend', label: 'Full outfit', icon: '✦' },
  { value: 'similar',   label: 'Find similar', icon: '◎' },
]

export default function ModeToggle({ mode, onChange }) {
  return (
    <div className="flex rounded-xl border border-white/10 overflow-hidden bg-white/[0.03]">
      {MODES.map((opt) => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={`
            flex items-center gap-2 px-5 py-2 text-sm font-medium transition-all duration-200
            ${mode === opt.value
              ? 'bg-white text-black'
              : 'text-white/40 hover:text-white/70 hover:bg-white/5'}
          `}
        >
          <span className="text-xs">{opt.icon}</span>
          {opt.label}
        </button>
      ))}
    </div>
  )
}

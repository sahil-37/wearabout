const OPTIONS = [
  { value: 'unisex', label: 'All' },
  { value: 'men',    label: 'Men' },
  { value: 'women',  label: 'Women' },
]

export default function GenderFilter({ value, onChange }) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-white/30 font-medium uppercase tracking-wider">Filter</span>
      <div className="flex rounded-xl border border-white/10 overflow-hidden bg-white/[0.03]">
        {OPTIONS.map((opt) => (
          <button
            key={opt.value}
            onClick={() => onChange(opt.value)}
            className={`
              px-4 py-1.5 text-sm font-medium transition-all duration-200
              ${value === opt.value
                ? 'bg-white text-black'
                : 'text-white/40 hover:text-white/70 hover:bg-white/5'}
            `}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  )
}

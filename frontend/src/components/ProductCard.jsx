import { useState } from 'react'

export default function ProductCard({ item }) {
  const [imgError, setImgError] = useState(false)
  const [hovered, setHovered] = useState(false)

  const score = Math.round(item.similarity_score * 100)

  return (
    <div
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className="group relative flex flex-col rounded-2xl overflow-hidden bg-[#111] border border-white/5 hover:border-white/15 transition-all duration-300 hover:-translate-y-1 hover:shadow-2xl hover:shadow-black/50"
    >
      {/* image */}
      <div className="relative overflow-hidden bg-[#1a1a1a] aspect-[3/4]">
        {!imgError ? (
          <img
            src={`/product-image?path=${encodeURIComponent(item.image_path)}`}
            alt={item.name}
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-5xl opacity-10">
            {categoryEmoji(item.category)}
          </div>
        )}

        {/* score badge */}
        <div className="absolute top-3 right-3">
          <span className={`
            text-xs font-bold px-2.5 py-1 rounded-full backdrop-blur-sm
            ${score >= 90 ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
            : score >= 75 ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
            : 'bg-white/10 text-white/50 border border-white/10'}
          `}>
            {score}%
          </span>
        </div>

        {/* category pill */}
        <div className="absolute top-3 left-3">
          <span className="text-xs px-2.5 py-1 rounded-full bg-black/50 backdrop-blur-sm text-white/60 border border-white/10 capitalize">
            {item.category}
          </span>
        </div>

        {/* hover overlay with buy button */}
        <div className={`
          absolute inset-0 flex items-end p-3 transition-all duration-300
          bg-gradient-to-t from-black/80 via-black/20 to-transparent
          ${hovered ? 'opacity-100' : 'opacity-0'}
        `}>
          {item.product_url && (
            <a
              href={item.product_url}
              target="_blank"
              rel="noopener noreferrer"
              onClick={(e) => e.stopPropagation()}
              className="w-full text-center text-sm font-semibold text-black bg-white hover:bg-white/90 rounded-xl py-2.5 transition-colors"
            >
              View on Myntra →
            </a>
          )}
        </div>
      </div>

      {/* info */}
      <div className="p-3 flex flex-col gap-1">
        <p className="text-xs text-white/35 uppercase tracking-widest font-medium">
          {item.brand}
        </p>
        <p className="text-sm text-white/80 font-medium leading-snug line-clamp-2">
          {item.name}
        </p>
        <p className="text-sm font-bold text-white mt-0.5">
          ₹{item.price?.toLocaleString('en-IN')}
        </p>
      </div>
    </div>
  )
}

function categoryEmoji(cat) {
  const map = {
    topwear: '👕', bottomwear: '👖',
    footwear: '👟', eyewear: '🕶️', handbag: '👜',
  }
  return map[cat?.toLowerCase()] ?? '👗'
}

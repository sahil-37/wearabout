export default function PreviewImage({ src }) {
  return (
    <div className="relative rounded-2xl overflow-hidden border border-white/8 shadow-2xl shadow-black/60">
      <img
        src={src}
        alt="Uploaded outfit"
        className="max-h-64 max-w-[180px] object-cover block"
      />
    </div>
  )
}

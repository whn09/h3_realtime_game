/** @type {import('next').NextConfig} */
const nextConfig = {
  // Nothing clever on purpose. The orchestrator serves both the API and the
  // clips, so there is no rewrite/proxy layer to get wrong -- the browser talks
  // to NEXT_PUBLIC_API_BASE directly and video URLs arrive absolute from the
  // API (`PUBLIC_BASE_URL` on that side), so they stay correct even once the
  // assets move behind a CDN.
  reactStrictMode: true,

  // Dev only, and not optional here: Next 16 refuses to serve `/_next/*` dev
  // resources to an origin it did not expect, and it only expects `localhost`.
  // Opening the page on `127.0.0.1` or the LAN address therefore got HTML with
  // no JavaScript behind it -- the page rendered, hydration never happened, and
  // every `useEffect` (including the one that loads the presets) silently never
  // ran. The only visible symptom was an empty presets grid.
  allowedDevOrigins: ["127.0.0.1", "192.168.31.240"],
};

export default nextConfig;

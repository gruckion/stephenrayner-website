import type { APIRoute } from "astro";

export const prerender = false;

export const GET: APIRoute = ({ request }) => {
  const timestamp = new Date().toISOString();
  const url = new URL(request.url);

  return new Response(
    JSON.stringify({
      status: "healthy",
      timestamp,
      environment: import.meta.env.MODE,
      url: url.origin,
      message: "Astro API route working on Cloudflare Pages"
    }),
    {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "no-cache, no-store, must-revalidate"
      }
    }
  );
};

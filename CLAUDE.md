# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development
```bash
pnpm dev       # Start development server
pnpm start     # Same as dev
```

### Build
```bash
pnpm build           # Build for production with remote DB
pnpm build-local     # Build for production with local DB
```

### Preview
```bash
pnpm preview   # Preview production build locally
```

### Astro CLI
```bash
pnpm astro     # Access Astro CLI commands
```

## Architecture

### Tech Stack
- **Framework**: Astro v4 with hybrid rendering (SSR + Static)
- **Deployment**: Cloudflare adapter
- **UI Components**: shadcn/ui with Radix UI primitives  
- **Styling**: Tailwind CSS with custom design tokens
- **Content**: MDX with Content Collections for blog, docs, guides, and releases
- **Forms**: React Hook Form with Zod validation, Simple Stack Form for newsletter
- **State**: React components for interactive features

### Project Structure
- `src/pages/` - Astro pages using file-based routing
- `src/components/` - React components including shadcn/ui components
- `src/layouts/` - Reusable Astro layouts (base, main, auth, blog, docs)
- `src/content/` - Content Collections for blog posts, releases, docs, guides
- `src/config/` - Site configuration and metadata
- `src/lib/` - Utilities including GraphQL client, fetchers, and helpers

### Key Patterns
- Pages use `.astro` files with frontmatter for metadata
- Interactive components use React with TypeScript
- Content uses MDX with syntax highlighting via Shiki
- Forms integrate with external services via Simple Stack Form
- API routes supported via Astro's hybrid output mode
- Path aliases configured via `@/*` pointing to `src/*`

### Component Architecture
- UI components follow shadcn/ui patterns with compound components
- Form components use controlled inputs with React Hook Form
- Theme switching via next-themes provider
- Mobile navigation via sheet component pattern
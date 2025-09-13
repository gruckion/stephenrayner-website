import React, { memo, useId, forwardRef, type ReactNode, type MouseEventHandler } from 'react';
import { motion, LazyMotion, domAnimation, type Variants } from 'framer-motion';
import { cn } from '@/lib/utils';

type Theme = 'light' | 'dark' | 'neutral';
type Variant = 'default' | 'subtle' | 'error';
type Size = 'sm' | 'default' | 'lg';
type IconVariant = 'left' | 'center' | 'right';

interface IconVariants {
  left: Variants;
  center: Variants;
  right: Variants;
}

const ICON_VARIANTS: IconVariants = {
  left: {
    initial: { scale: 0.8, opacity: 0, x: 0, y: 0, rotate: 0 },
    animate: { scale: 1, opacity: 1, x: 0, y: 0, rotate: -6, transition: { duration: 0.4, delay: 0.1 } },
    hover: { x: -22, y: -5, rotate: -15, scale: 1.1, transition: { duration: 0.2 } }
  },
  center: {
    initial: { scale: 0.8, opacity: 0 },
    animate: { scale: 1, opacity: 1, transition: { duration: 0.4, delay: 0.2 } },
    hover: { y: -10, scale: 1.15, transition: { duration: 0.2 } }
  },
  right: {
    initial: { scale: 0.8, opacity: 0, x: 0, y: 0, rotate: 0 },
    animate: { scale: 1, opacity: 1, x: 0, y: 0, rotate: 6, transition: { duration: 0.4, delay: 0.3 } },
    hover: { x: 22, y: -5, rotate: 15, scale: 1.1, transition: { duration: 0.2 } }
  }
};

const CONTENT_VARIANTS: Variants = {
  initial: { y: 20, opacity: 0 },
  animate: { y: 0, opacity: 1, transition: { duration: 0.4, delay: 0.2 } },
};

const BUTTON_VARIANTS: Variants = {
  initial: { y: 20, opacity: 0 },
  animate: { y: 0, opacity: 1, transition: { duration: 0.4, delay: 0.3 } },
};

interface IconContainerProps {
  children: ReactNode;
  variant: IconVariant;
  className?: string;
  theme: Theme;
}

const IconContainer = memo<IconContainerProps>(({ children, variant, className = '', theme }) => (
  <motion.div
    variants={ICON_VARIANTS[variant]}
    className={cn(
      "w-12 h-12 rounded-xl flex items-center justify-center relative shadow-lg transition-all duration-300",
      theme === 'dark' && "bg-neutral-800 border border-neutral-700 group-hover:shadow-xl group-hover:border-neutral-600",
      theme === 'neutral' && "bg-stone-100 border border-stone-200 group-hover:shadow-xl group-hover:border-stone-300",
      (!theme || theme === 'light') && "bg-white border border-gray-200 group-hover:shadow-xl group-hover:border-gray-300",
      className
    )}
  >
    <div className={cn(
      "text-sm transition-colors duration-300",
      theme === 'dark' && "text-neutral-400 group-hover:text-neutral-200",
      theme === 'neutral' && "text-stone-500 group-hover:text-stone-700",
      (!theme || theme === 'light') && "text-gray-500 group-hover:text-gray-700"
    )}>
      {children}
    </div>
  </motion.div>
));
IconContainer.displayName = "IconContainer";

interface MultiIconDisplayProps {
  icons: ReactNode[];
  theme: Theme;
}

const MultiIconDisplay = memo<MultiIconDisplayProps>(({ icons, theme }) => {
  if (!icons || icons.length < 3) return null;

  return (
    <div className="flex justify-center isolate relative">
      <IconContainer variant="left" className="left-2 top-1 z-10" theme={theme}>
        {icons[0]}
      </IconContainer>
      <IconContainer variant="center" className="z-20" theme={theme}>
        {icons[1]}
      </IconContainer>
      <IconContainer variant="right" className="right-2 top-1 z-10" theme={theme}>
        {icons[2]}
      </IconContainer>
    </div>
  );
});
MultiIconDisplay.displayName = "MultiIconDisplay";

interface BackgroundProps {
  theme: Theme;
}

const Background: React.FC<BackgroundProps> = () => (
  <div
    aria-hidden="true"
    className="absolute inset-0 opacity-0 group-hover:opacity-[0.02] transition-opacity duration-500"
    style={{
      backgroundImage: `radial-gradient(circle at 2px 2px, #fff 1px, transparent 1px)`,
      backgroundSize: '24px 24px'
    }}
  />
);

interface EmptyStateAction {
  label: string;
  onClick: MouseEventHandler<HTMLButtonElement>;
  icon?: ReactNode;
  disabled?: boolean;
}

interface EmptyStateProps {
  title: string;
  description?: string;
  icons?: ReactNode[];
  action?: EmptyStateAction;
  variant?: Variant;
  size?: Size;
  theme?: Theme;
  isIconAnimated?: boolean;
  className?: string;
}

type VariantClasses = Record<Variant, Record<Theme, string>>;
type SizeClasses = Record<Size, string>;
type TextSizeClasses = Record<'title' | 'description', Record<Size, string>>;
type TextColorClasses = Record<'title' | 'description', Record<Theme, string>>;

export const EmptyState = forwardRef<HTMLElement, EmptyStateProps>(({
  title,
  description,
  icons,
  action,
  variant = 'default',
  size = 'default',
  theme = 'light',
  isIconAnimated = true,
  className = '',
}, ref) => {
  const titleId = useId();
  const descriptionId = useId();

  const baseClasses = "group transition-all duration-300 rounded-xl relative overflow-hidden text-center flex flex-col items-center justify-center";

  const sizeClasses: SizeClasses = {
    sm: "p-6",
    default: "p-8",
    lg: "p-12"
  };

  const getVariantClasses = (variant: Variant, theme: Theme): string => {
    const variants: VariantClasses = {
      default: {
        light: "bg-white border-dashed border-2 border-gray-300 hover:border-gray-400 hover:bg-gray-50/50",
        dark: "bg-neutral-900 border-dashed border-2 border-neutral-700 hover:border-neutral-600 hover:bg-neutral-800/50",
        neutral: "bg-stone-50 border-dashed border-2 border-stone-300 hover:border-stone-400 hover:bg-stone-100/50"
      },
      subtle: {
        light: "bg-white border border-transparent hover:bg-gray-50/30",
        dark: "bg-neutral-900 border border-transparent hover:bg-neutral-800/30",
        neutral: "bg-stone-50 border border-transparent hover:bg-stone-100/30"
      },
      error: {
        light: "bg-white border border-red-200 bg-red-50/50 hover:bg-red-50/80",
        dark: "bg-neutral-900 border border-red-800 bg-red-950/50 hover:bg-red-950/80",
        neutral: "bg-stone-50 border border-red-300 bg-red-50/50 hover:bg-red-50/80"
      }
    };
    return variants[variant][theme];
  };

  const getTextClasses = (type: 'title' | 'description', size: Size, theme: Theme): string => {
    const sizes: TextSizeClasses = {
      title: {
        sm: "text-base",
        default: "text-lg",
        lg: "text-xl"
      },
      description: {
        sm: "text-xs",
        default: "text-sm",
        lg: "text-base"
      }
    };

    const colors: TextColorClasses = {
      title: {
        light: "text-gray-900",
        dark: "text-neutral-100",
        neutral: "text-stone-900"
      },
      description: {
        light: "text-gray-600",
        dark: "text-neutral-400",
        neutral: "text-stone-600"
      }
    };

    return cn(sizes[type][size], colors[type][theme], "font-semibold transition-colors duration-200");
  };

  const getButtonClasses = (size: Size, theme: Theme): string => {
    const sizeClasses: SizeClasses = {
      sm: "text-xs px-3 py-1.5",
      default: "text-sm px-4 py-2",
      lg: "text-base px-6 py-3"
    };

    const themeClasses: Record<Theme, string> = {
      light: "border-gray-300 bg-white hover:bg-gray-50 text-gray-700",
      dark: "border-neutral-600 bg-neutral-800 hover:bg-neutral-700 text-neutral-200",
      neutral: "border-stone-300 bg-stone-100 hover:bg-stone-200 text-stone-700"
    };

    return cn(
      "inline-flex items-center gap-2 border rounded-md font-medium shadow-sm hover:shadow-md transition-all duration-200 relative overflow-hidden group/button disabled:opacity-50 disabled:cursor-not-allowed",
      sizeClasses[size],
      themeClasses[theme]
    );
  };

  return (
    <LazyMotion features={domAnimation}>
      <motion.section
        ref={ref}
        role="region"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        className={cn(
          baseClasses,
          sizeClasses[size],
          getVariantClasses(variant, theme),
          className
        )}
        initial="initial"
        animate="animate"
        whileHover={isIconAnimated ? "hover" : "animate"}
      >
        <Background theme={theme} />
        <div className="relative z-10 flex flex-col items-center">
          {icons && icons.length >= 3 && (
            <div className="mb-6">
              <MultiIconDisplay icons={icons} theme={theme} />
            </div>
          )}

          <motion.div variants={CONTENT_VARIANTS} className="space-y-2 mb-6">
            <h2 id={titleId} className={getTextClasses('title', size, theme)}>
              {title}
            </h2>
            {description && (
              <p
                id={descriptionId}
                className={cn(
                  getTextClasses('description', size, theme).replace('font-semibold', ''),
                  "max-w-md leading-relaxed"
                )}
              >
                {description}
              </p>
            )}
          </motion.div>

          {action && (
            <motion.div variants={BUTTON_VARIANTS}>
              <motion.button
                type="button"
                onClick={action.onClick}
                disabled={action.disabled}
                className={getButtonClasses(size, theme)}
                whileTap={{ scale: 0.98 }}
              >
                {action.icon && (
                  <motion.div
                    className="transition-transform group-hover/button:rotate-90"
                    whileHover={{ rotate: 90 }}
                  >
                    {action.icon}
                  </motion.div>
                )}
                <span className="relative z-10">{action.label}</span>
              </motion.button>
            </motion.div>
          )}
        </div>
      </motion.section>
    </LazyMotion>
  );
});
EmptyState.displayName = "EmptyState";

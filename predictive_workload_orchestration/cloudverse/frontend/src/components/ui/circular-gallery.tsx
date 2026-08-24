import { forwardRef, useEffect, useMemo, useRef, useState } from 'react';
import type { CSSProperties, HTMLAttributes } from 'react';

export interface GalleryItem {
  id: string;
  name: string;
  role: string;
  image: string;
  imageAlt: string;
  imagePosition?: string;
}

interface CircularGalleryProps extends HTMLAttributes<HTMLDivElement> {
  items: GalleryItem[];
  selectedId?: string | null;
  radius?: number;
  autoRotateSpeed?: number;
  onItemSelect?: (id: string) => void;
}

export const CircularGallery = forwardRef<HTMLDivElement, CircularGalleryProps>(
  (
    {
      items,
      selectedId,
      radius = 600,
      autoRotateSpeed = 0.025,
      onItemSelect,
      className = '',
      ...props
    },
    ref
  ) => {
    const [rotation, setRotation] = useState(0);
    const [isScrolling, setIsScrolling] = useState(false);
    const [viewportWidth, setViewportWidth] = useState(
      typeof window === 'undefined' ? 1440 : window.innerWidth
    );
    const rootRef = useRef<HTMLDivElement | null>(null);
    const scrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const animationFrameRef = useRef<number | null>(null);

    const mergedRef = useMemo(() => {
      return (node: HTMLDivElement | null) => {
        rootRef.current = node;
        if (typeof ref === 'function') {
          ref(node);
        } else if (ref) {
          ref.current = node;
        }
      };
    }, [ref]);

    const responsiveRadius = Math.min(radius, Math.max(280, viewportWidth * 0.42));

    useEffect(() => {
      const handleScroll = () => {
        setIsScrolling(true);

        if (scrollTimeoutRef.current) {
          clearTimeout(scrollTimeoutRef.current);
        }

        const scrollSection = rootRef.current?.closest('.technology-gallery-scroll');
        const rect = scrollSection?.getBoundingClientRect();
        const scrollableHeight = rect ? rect.height - window.innerHeight : document.documentElement.scrollHeight - window.innerHeight;
        const rawProgress = rect ? -rect.top / scrollableHeight : window.scrollY / scrollableHeight;
        const scrollProgress = Math.max(0, Math.min(1, scrollableHeight > 0 ? rawProgress : 0));
        setRotation(scrollProgress * 360);

        scrollTimeoutRef.current = setTimeout(() => setIsScrolling(false), 180);
      };

      window.addEventListener('scroll', handleScroll, { passive: true });

      return () => {
        window.removeEventListener('scroll', handleScroll);
        if (scrollTimeoutRef.current) {
          clearTimeout(scrollTimeoutRef.current);
        }
      };
    }, []);

    useEffect(() => {
      const handleResize = () => setViewportWidth(window.innerWidth);

      handleResize();
      window.addEventListener('resize', handleResize);

      return () => window.removeEventListener('resize', handleResize);
    }, []);

    useEffect(() => {
      const autoRotate = () => {
        if (!isScrolling && !selectedId) {
          setRotation((previous) => previous + autoRotateSpeed);
        }

        animationFrameRef.current = requestAnimationFrame(autoRotate);
      };

      animationFrameRef.current = requestAnimationFrame(autoRotate);

      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      };
    }, [autoRotateSpeed, isScrolling, selectedId]);

    const anglePerItem = 360 / items.length;

    return (
      <div
        ref={mergedRef}
        role="region"
        aria-label="Circular technology gallery"
        className={`circular-gallery ${className}`}
        {...props}
      >
        <div
          className="circular-gallery-track"
          style={{
            transform: `rotateY(${rotation}deg)`,
          }}
        >
          {items.map((item, index) => {
            const itemAngle = index * anglePerItem;
            const relativeAngle = (itemAngle + rotation + 360) % 360;
            const normalizedAngle = Math.abs(relativeAngle > 180 ? 360 - relativeAngle : relativeAngle);
            const opacity = Math.max(0.34, 1 - normalizedAngle / 180);
            const isSelected = selectedId === item.id;

            return (
              <button
                key={item.id}
                type="button"
                className={`circular-gallery-card ${isSelected ? 'selected' : ''}`}
                aria-label={`Open details for ${item.name}`}
                onClick={() => onItemSelect?.(item.id)}
                style={
                  {
                    '--item-opacity': opacity,
                    '--item-angle': `${itemAngle}deg`,
                    '--item-radius': `${responsiveRadius}px`,
                  } as CSSProperties
                }
              >
                <img
                  src={item.image}
                  alt={item.imageAlt}
                  style={{ objectPosition: item.imagePosition || 'center' }}
                />
                <div className="gallery-card-caption">
                  <strong>{item.name}</strong>
                  <span>{item.role}</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>
    );
  }
);

CircularGallery.displayName = 'CircularGallery';

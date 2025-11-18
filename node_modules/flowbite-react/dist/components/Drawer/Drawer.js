'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef, useId, useEffect } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { DrawerContext } from './DrawerContext.js';
import { drawerTheme } from './theme.js';

const Drawer = forwardRef((props, ref) => {
  const id = useId();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [drawerTheme, provider.theme?.drawer, props.theme],
    [get(provider.clearTheme, "drawer"), props.clearTheme],
    [get(provider.applyTheme, "drawer"), props.applyTheme]
  );
  const {
    backdrop = true,
    children,
    className,
    edge = false,
    position = "left",
    onClose,
    open: isOpen = false,
    ...restProps
  } = resolveProps(props, provider.props?.drawer);
  useEffect(() => {
    const handleEscapeKey = (event) => {
      if (event.key === "Escape" && isOpen && onClose) {
        onClose();
      }
    };
    document.addEventListener("keydown", handleEscapeKey);
    return () => document.removeEventListener("keydown", handleEscapeKey);
  }, [onClose, isOpen]);
  return /* @__PURE__ */ jsxs(
    DrawerContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, onClose, isOpen, id },
      children: [
        /* @__PURE__ */ jsx(
          "div",
          {
            ref,
            "aria-modal": true,
            "aria-describedby": `drawer-dialog-${id}`,
            role: "dialog",
            tabIndex: -1,
            "data-testid": "flowbite-drawer",
            className: twMerge(
              theme.root.base,
              theme.root.position[position][isOpen ? "on" : "off"],
              edge && !isOpen && theme.root.edge,
              className
            ),
            ...restProps,
            children
          }
        ),
        isOpen && backdrop && /* @__PURE__ */ jsx("div", { onClick: () => onClose(), className: theme.root.backdrop })
      ]
    }
  );
});
Drawer.displayName = "Drawer";

export { Drawer };
//# sourceMappingURL=Drawer.js.map

'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var DrawerContext = require('./DrawerContext.cjs');
var theme = require('./theme.cjs');

const Drawer = React.forwardRef((props, ref) => {
  const id = React.useId();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.drawerTheme, provider$1.theme?.drawer, props.theme],
    [get.get(provider$1.clearTheme, "drawer"), props.clearTheme],
    [get.get(provider$1.applyTheme, "drawer"), props.applyTheme]
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
  } = resolveProps.resolveProps(props, provider$1.props?.drawer);
  React.useEffect(() => {
    const handleEscapeKey = (event) => {
      if (event.key === "Escape" && isOpen && onClose) {
        onClose();
      }
    };
    document.addEventListener("keydown", handleEscapeKey);
    return () => document.removeEventListener("keydown", handleEscapeKey);
  }, [onClose, isOpen]);
  return /* @__PURE__ */ jsxRuntime.jsxs(
    DrawerContext.DrawerContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, onClose, isOpen, id },
      children: [
        /* @__PURE__ */ jsxRuntime.jsx(
          "div",
          {
            ref,
            "aria-modal": true,
            "aria-describedby": `drawer-dialog-${id}`,
            role: "dialog",
            tabIndex: -1,
            "data-testid": "flowbite-drawer",
            className: tailwindMerge.twMerge(
              theme$1.root.base,
              theme$1.root.position[position][isOpen ? "on" : "off"],
              edge && !isOpen && theme$1.root.edge,
              className
            ),
            ...restProps,
            children
          }
        ),
        isOpen && backdrop && /* @__PURE__ */ jsxRuntime.jsx("div", { onClick: () => onClose(), className: theme$1.root.backdrop })
      ]
    }
  );
});
Drawer.displayName = "Drawer";

exports.Drawer = Drawer;
//# sourceMappingURL=Drawer.cjs.map

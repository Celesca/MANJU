'use client';
'use strict';

var jsxRuntime = require('react/jsx-runtime');
var React = require('react');
var get = require('../../helpers/get.cjs');
var resolveProps = require('../../helpers/resolve-props.cjs');
var resolveTheme = require('../../helpers/resolve-theme.cjs');
var tailwindMerge = require('../../helpers/tailwind-merge.cjs');
var provider = require('../../theme/provider.cjs');
var theme = require('./theme.cjs');

const Tabs = React.forwardRef((props, ref) => {
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.tabsTheme, provider$1.theme?.tabs, props.theme],
    [get.get(provider$1.clearTheme, "tabs"), props.clearTheme],
    [get.get(provider$1.applyTheme, "tabs"), props.applyTheme]
  );
  const {
    children,
    className,
    onActiveTabChange,
    variant = "default",
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.tabs);
  const id = React.useId();
  const tabs = React.useMemo(
    () => React.Children.map(React.Children.toArray(children), ({ props: props2 }) => props2),
    [children]
  );
  const tabRefs = React.useRef([]);
  const [activeTab, setActiveTab] = React.useState(
    Math.max(
      0,
      tabs.findIndex((tab) => tab.active)
    )
  );
  const [focusedTab, setFocusedTab] = React.useState(-1);
  function setActiveTabWithCallback(activeTab2) {
    setActiveTab(activeTab2);
    if (onActiveTabChange) onActiveTabChange(activeTab2);
  }
  function handleClick({ target }) {
    setActiveTabWithCallback(target);
    setFocusedTab(target);
  }
  function handleKeyboard({ event, target }) {
    if (event.key === "ArrowLeft") {
      setFocusedTab(Math.max(0, focusedTab - 1));
    }
    if (event.key === "ArrowRight") {
      setFocusedTab(Math.min(tabs.length - 1, focusedTab + 1));
    }
    if (event.key === "Enter") {
      setActiveTabWithCallback(target);
      setFocusedTab(target);
    }
  }
  const tabItemStyle = theme$1.tablist.tabitem.variant[variant];
  const tabItemContainerStyle = theme$1.tabitemcontainer.variant[variant];
  React.useEffect(() => {
    tabRefs.current[focusedTab]?.focus();
  }, [focusedTab]);
  React.useImperativeHandle(ref, () => ({
    setActiveTab: setActiveTabWithCallback
  }));
  return /* @__PURE__ */ jsxRuntime.jsxs("div", { className: tailwindMerge.twMerge(theme$1.base, className), children: [
    /* @__PURE__ */ jsxRuntime.jsx(
      "div",
      {
        "aria-label": "Tabs",
        role: "tablist",
        className: tailwindMerge.twMerge(theme$1.tablist.base, theme$1.tablist.variant[variant], className),
        ...restProps,
        children: tabs.map((tab, index) => /* @__PURE__ */ jsxRuntime.jsxs(
          "button",
          {
            type: "button",
            "aria-controls": `${id}-tabpanel-${index}`,
            "aria-selected": index === activeTab,
            className: tailwindMerge.twMerge(
              theme$1.tablist.tabitem.base,
              tabItemStyle.base,
              index === activeTab && tabItemStyle.active.on,
              index !== activeTab && !tab.disabled && tabItemStyle.active.off
            ),
            disabled: tab.disabled,
            id: `${id}-tab-${index}`,
            onClick: () => handleClick({ target: index }),
            onKeyDown: (event) => handleKeyboard({ event, target: index }),
            ref: (element) => {
              tabRefs.current[index] = element;
            },
            role: "tab",
            tabIndex: index === focusedTab ? 0 : -1,
            style: { zIndex: index === focusedTab ? 2 : 1 },
            children: [
              tab.icon && /* @__PURE__ */ jsxRuntime.jsx(tab.icon, { className: theme$1.tablist.tabitem.icon }),
              tab.title
            ]
          },
          index
        ))
      }
    ),
    /* @__PURE__ */ jsxRuntime.jsx("div", { className: tailwindMerge.twMerge(theme$1.tabitemcontainer.base, tabItemContainerStyle), children: tabs.map((tab, index) => /* @__PURE__ */ jsxRuntime.jsx(
      "div",
      {
        "aria-labelledby": `${id}-tab-${index}`,
        className: theme$1.tabpanel,
        hidden: index !== activeTab,
        id: `${id}-tabpanel-${index}`,
        role: "tabpanel",
        tabIndex: 0,
        children: tab.children
      },
      index
    )) })
  ] });
});
Tabs.displayName = "Tabs";

exports.Tabs = Tabs;
//# sourceMappingURL=Tabs.cjs.map

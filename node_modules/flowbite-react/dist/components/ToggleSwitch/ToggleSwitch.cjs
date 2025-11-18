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

const ToggleSwitch = React.forwardRef((props, ref) => {
  const id = React.useId();
  const provider$1 = provider.useThemeProvider();
  const theme$1 = resolveTheme.useResolveTheme(
    [theme.toggleSwitchTheme, provider$1.theme?.toggleSwitch, props.theme],
    [get.get(provider$1.clearTheme, "toggleSwitch"), props.clearTheme],
    [get.get(provider$1.applyTheme, "toggleSwitch"), props.applyTheme]
  );
  const {
    checked,
    className,
    color = "default",
    sizing = "md",
    disabled,
    label,
    name,
    onChange,
    ...restProps
  } = resolveProps.resolveProps(props, provider$1.props?.toggleSwitch);
  function handleClick() {
    onChange(!checked);
  }
  function handleOnKeyDown(event) {
    if (event.code == "Enter") {
      event.preventDefault();
    }
  }
  return /* @__PURE__ */ jsxRuntime.jsxs(jsxRuntime.Fragment, { children: [
    /* @__PURE__ */ jsxRuntime.jsx("input", { ref, checked, name, type: "checkbox", className: theme$1.root.input, readOnly: true, hidden: true }),
    /* @__PURE__ */ jsxRuntime.jsxs(
      "button",
      {
        "aria-checked": checked,
        "aria-labelledby": `${id}-flowbite-toggleswitch-label`,
        disabled,
        id: `${id}-flowbite-toggleswitch`,
        onClick: handleClick,
        onKeyDown: handleOnKeyDown,
        role: "switch",
        tabIndex: 0,
        type: "button",
        className: tailwindMerge.twMerge(theme$1.root.base, theme$1.root.active[disabled ? "off" : "on"], className),
        ...restProps,
        children: [
          /* @__PURE__ */ jsxRuntime.jsx(
            "div",
            {
              "data-testid": "flowbite-toggleswitch-toggle",
              className: tailwindMerge.twMerge(
                theme$1.toggle.base,
                theme$1.toggle.checked.color[color],
                theme$1.toggle.checked[checked ? "on" : "off"],
                theme$1.toggle.sizes[sizing]
              )
            }
          ),
          !!label?.length && /* @__PURE__ */ jsxRuntime.jsx(
            "span",
            {
              "data-testid": "flowbite-toggleswitch-label",
              id: `${id}-flowbite-toggleswitch-label`,
              className: theme$1.root.label,
              children: label
            }
          )
        ]
      }
    )
  ] });
});
ToggleSwitch.displayName = "ToggleSwitch";

exports.ToggleSwitch = ToggleSwitch;
//# sourceMappingURL=ToggleSwitch.cjs.map

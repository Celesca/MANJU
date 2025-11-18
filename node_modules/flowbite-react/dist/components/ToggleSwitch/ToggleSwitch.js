'use client';
import { jsxs, Fragment, jsx } from 'react/jsx-runtime';
import { forwardRef, useId } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { toggleSwitchTheme } from './theme.js';

const ToggleSwitch = forwardRef((props, ref) => {
  const id = useId();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [toggleSwitchTheme, provider.theme?.toggleSwitch, props.theme],
    [get(provider.clearTheme, "toggleSwitch"), props.clearTheme],
    [get(provider.applyTheme, "toggleSwitch"), props.applyTheme]
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
  } = resolveProps(props, provider.props?.toggleSwitch);
  function handleClick() {
    onChange(!checked);
  }
  function handleOnKeyDown(event) {
    if (event.code == "Enter") {
      event.preventDefault();
    }
  }
  return /* @__PURE__ */ jsxs(Fragment, { children: [
    /* @__PURE__ */ jsx("input", { ref, checked, name, type: "checkbox", className: theme.root.input, readOnly: true, hidden: true }),
    /* @__PURE__ */ jsxs(
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
        className: twMerge(theme.root.base, theme.root.active[disabled ? "off" : "on"], className),
        ...restProps,
        children: [
          /* @__PURE__ */ jsx(
            "div",
            {
              "data-testid": "flowbite-toggleswitch-toggle",
              className: twMerge(
                theme.toggle.base,
                theme.toggle.checked.color[color],
                theme.toggle.checked[checked ? "on" : "off"],
                theme.toggle.sizes[sizing]
              )
            }
          ),
          !!label?.length && /* @__PURE__ */ jsx(
            "span",
            {
              "data-testid": "flowbite-toggleswitch-label",
              id: `${id}-flowbite-toggleswitch-label`,
              className: theme.root.label,
              children: label
            }
          )
        ]
      }
    )
  ] });
});
ToggleSwitch.displayName = "ToggleSwitch";

export { ToggleSwitch };
//# sourceMappingURL=ToggleSwitch.js.map

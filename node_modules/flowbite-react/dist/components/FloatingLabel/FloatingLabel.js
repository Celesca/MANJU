'use client';
import { jsxs, jsx } from 'react/jsx-runtime';
import { forwardRef, useId } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { floatingLabelTheme } from './theme.js';

const FloatingLabel = forwardRef((props, ref) => {
  const randomId = useId();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [floatingLabelTheme, provider.theme?.floatingLabel, props.theme],
    [get(provider.clearTheme, "floatingLabel"), props.clearTheme],
    [get(provider.applyTheme, "floatingLabel"), props.applyTheme]
  );
  const {
    label,
    color = "default",
    sizing = "md",
    variant,
    disabled = false,
    className,
    ...restProps
  } = resolveProps(props, provider.props?.floatingLabel);
  return /* @__PURE__ */ jsxs("div", { className: twMerge("relative", variant === "standard" ? "z-0" : ""), children: [
    /* @__PURE__ */ jsx(
      "input",
      {
        type: "text",
        id: "floatingLabel" + randomId,
        "aria-describedby": "outlined_success_help",
        className: twMerge(theme.input[color][variant][sizing], className),
        placeholder: " ",
        "data-testid": "floating-label",
        disabled,
        ...restProps,
        ref
      }
    ),
    /* @__PURE__ */ jsx("label", { htmlFor: "floatingLabel" + randomId, className: twMerge(theme.label[color][variant][sizing], className), children: label })
  ] });
});
FloatingLabel.displayName = "FloatingLabel";

export { FloatingLabel };
//# sourceMappingURL=FloatingLabel.js.map

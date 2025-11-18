'use client';
import { jsx, jsxs } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { TableContext } from './TableContext.js';
import { tableTheme } from './theme.js';

const Table = forwardRef((props, ref) => {
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [tableTheme, provider.theme?.table, props.theme],
    [get(provider.clearTheme, "table"), props.clearTheme],
    [get(provider.applyTheme, "table"), props.applyTheme]
  );
  const { className, striped, hoverable, ...restProps } = resolveProps(props, provider.props?.table);
  return /* @__PURE__ */ jsx("div", { "data-testid": "table-element", className: twMerge(theme.root.wrapper), children: /* @__PURE__ */ jsxs(
    TableContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme, striped, hoverable },
      children: [
        /* @__PURE__ */ jsx("div", { className: twMerge(theme.root.shadow, className) }),
        /* @__PURE__ */ jsx("table", { ref, className: twMerge(theme.root.base, className), ...restProps })
      ]
    }
  ) });
});
Table.displayName = "Table";

export { Table };
//# sourceMappingURL=Table.js.map

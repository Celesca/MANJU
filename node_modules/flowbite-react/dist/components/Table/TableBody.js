'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { TableBodyContext } from './TableBodyContext.js';
import { useTableContext } from './TableContext.js';
import { tableTheme } from './theme.js';

const TableBody = forwardRef((props, ref) => {
  const { theme: rootTheme, clearTheme: rootClearTheme, applyTheme: rootApplyTheme } = useTableContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [tableTheme.body, provider.theme?.table?.body, rootTheme?.body, props.theme],
    [get(provider.clearTheme, "table.body"), get(rootClearTheme, "body"), props.clearTheme],
    [get(provider.applyTheme, "table.body"), get(rootApplyTheme, "body"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.tableBody);
  return /* @__PURE__ */ jsx(
    TableBodyContext.Provider,
    {
      value: { theme: props.theme, clearTheme: props.clearTheme, applyTheme: props.applyTheme },
      children: /* @__PURE__ */ jsx("tbody", { ref, className: twMerge(theme.base, className), ...restProps })
    }
  );
});
TableBody.displayName = "TableBody";

export { TableBody };
//# sourceMappingURL=TableBody.js.map

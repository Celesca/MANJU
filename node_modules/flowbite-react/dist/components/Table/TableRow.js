'use client';
import { jsx } from 'react/jsx-runtime';
import { forwardRef } from 'react';
import { get } from '../../helpers/get.js';
import { resolveProps } from '../../helpers/resolve-props.js';
import { useResolveTheme } from '../../helpers/resolve-theme.js';
import { twMerge } from '../../helpers/tailwind-merge.js';
import { useThemeProvider } from '../../theme/provider.js';
import { useTableContext } from './TableContext.js';
import { tableTheme } from './theme.js';

const TableRow = forwardRef((props, ref) => {
  const {
    theme: rootTheme,
    clearTheme: rootClearTheme,
    applyTheme: rootApplyTheme,
    hoverable,
    striped
  } = useTableContext();
  const provider = useThemeProvider();
  const theme = useResolveTheme(
    [tableTheme.row, provider.theme?.table?.row, rootTheme?.row, props.theme],
    [get(provider.clearTheme, "table.row"), get(rootClearTheme, "row"), props.clearTheme],
    [get(provider.applyTheme, "table.row"), get(rootApplyTheme, "row"), props.applyTheme]
  );
  const { className, ...restProps } = resolveProps(props, provider.props?.tableRow);
  return /* @__PURE__ */ jsx(
    "tr",
    {
      ref,
      "data-testid": "table-row-element",
      className: twMerge(theme.base, striped && theme.striped, hoverable && theme.hovered, className),
      ...restProps
    }
  );
});
TableRow.displayName = "TableRow";

export { TableRow };
//# sourceMappingURL=TableRow.js.map

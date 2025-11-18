import { type ComponentPropsWithRef } from "react";
import type { ThemingProps } from "../../types";
export interface TableRowTheme {
    base: string;
    hovered: string;
    striped: string;
}
export interface TableRowProps extends ComponentPropsWithRef<"tr">, ThemingProps<TableRowTheme> {
}
export declare const TableRow: import("react").ForwardRefExoticComponent<Omit<TableRowProps, "ref"> & import("react").RefAttributes<HTMLTableRowElement>>;

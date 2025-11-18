import { type ComponentPropsWithRef } from "react";
import type { ThemingProps } from "../../types";
import type { TableBodyTheme } from "./TableBody";
import type { TableHeadTheme } from "./TableHead";
import type { TableRowTheme } from "./TableRow";
export interface TableTheme {
    root: TableRootTheme;
    head: TableHeadTheme;
    row: TableRowTheme;
    body: TableBodyTheme;
}
export interface TableRootTheme {
    base: string;
    shadow: string;
    wrapper: string;
}
export interface TableProps extends ComponentPropsWithRef<"table">, ThemingProps<TableTheme> {
    striped?: boolean;
    hoverable?: boolean;
}
export declare const Table: import("react").ForwardRefExoticComponent<Omit<TableProps, "ref"> & import("react").RefAttributes<HTMLTableElement>>;

import { type ComponentPropsWithRef } from "react";
import type { ThemingProps } from "../../types";
export interface TableHeadCellTheme {
    base: string;
}
export interface TableHeadCellProps extends ComponentPropsWithRef<"th">, ThemingProps<TableHeadCellTheme> {
}
export declare const TableHeadCell: import("react").ForwardRefExoticComponent<Omit<TableHeadCellProps, "ref"> & import("react").RefAttributes<HTMLTableCellElement>>;

import type { ThemingProps } from "../../types";
import type { TableBodyTheme } from "./TableBody";
export type TableBodyContextValue = ThemingProps<TableBodyTheme>;
export declare const TableBodyContext: import("react").Context<TableBodyContextValue | undefined>;
export declare function useTableBodyContext(): TableBodyContextValue;

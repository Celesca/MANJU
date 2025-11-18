import type { ThemingProps } from "../../types";
import type { TableTheme } from "./Table";
export interface TableContextValue extends ThemingProps<TableTheme> {
    striped?: boolean;
    hoverable?: boolean;
}
export declare const TableContext: import("react").Context<TableContextValue | undefined>;
export declare function useTableContext(): TableContextValue;

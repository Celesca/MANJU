import type { DynamicStringEnumKeysOf, ThemingProps } from "../../types";
import type { RatingTheme } from "./Rating";
import type { RatingStarSizes } from "./RatingStar";
export interface RatingContextValue extends ThemingProps<RatingTheme> {
    size?: DynamicStringEnumKeysOf<RatingStarSizes>;
}
export declare const RatingContext: import("react").Context<RatingContextValue | undefined>;
export declare function useRatingContext(): RatingContextValue;
